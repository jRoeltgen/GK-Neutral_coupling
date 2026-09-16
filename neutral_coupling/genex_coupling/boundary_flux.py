"""Conservative particle-flux transfer between unlike structured meshes.

The routines here deliberately describe boundaries by geometry and connectivity,
not by equilibrium labels (inner target, PFR, and so on).  That keeps the
normalisation independent of the number of divertor legs and SOLPS cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BoundaryComponent:
    """A connected run of cell-centred samples representing boundary faces."""

    name: str
    kind: str                 # ``poloidal`` or ``radial``
    indices: np.ndarray       # (n, 2) indices into the owning field
    sign: float               # field-coordinate sign which points outwards
    centres: np.ndarray       # (n, 2), columns R and Z [m]
    areas: np.ndarray         # axisymmetric face areas [m2]


def axisymmetric_point_areas(centres):
    """Return ``2*pi*R*dl`` control areas for ordered face-centre samples."""
    points = np.asarray(centres, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("face centres must have shape (n, 2)")
    if len(points) == 0:
        return np.empty(0)
    widths = np.ones(1) if len(points) == 1 else np.empty(len(points))
    if len(points) > 1:
        segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
        # Native GENE-X samples represent cells, not quadrature points at the
        # ends of a prescribed line. Extrapolate half a cell at both ends.
        widths[0] = segments[0]
        widths[-1] = segments[-1]
        if len(points) > 2:
            widths[1:-1] = (segments[:-1] + segments[1:]) / 2
    return 2 * np.pi * points[:, 0] * widths


def _runs(indices, varying_axis):
    """Split boundary indices into contiguous logical runs."""
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return []
    fixed_axis = 1 - varying_axis
    order = np.lexsort((indices[:, varying_axis], indices[:, fixed_axis]))
    values = indices[order]
    breaks = np.where(
        (np.diff(values[:, fixed_axis]) != 0)
        | (np.diff(values[:, varying_axis]) != 1)
    )[0] + 1
    return [part for part in np.split(values, breaks) if len(part)]


def _structured_areas(indices, centres, r, z, varying_axis):
    if len(indices) > 1:
        return axisymmetric_point_areas(centres)
    index = indices[0].copy()
    distances = []
    for step in (-1, 1):
        other = index.copy()
        other[varying_axis] += step
        if 0 <= other[varying_axis] < r.shape[varying_axis]:
            distances.append(np.hypot(
                r[tuple(other)] - centres[0, 0],
                z[tuple(other)] - centres[0, 1],
            ))
    if not distances:
        raise ValueError("cannot infer area of an isolated one-cell grid")
    return np.array([2 * np.pi * centres[0, 0] * min(distances)])


def structured_boundary_components(active, r, z, *, target=None, prefix="mesh"):
    """Extract topology-free boundary runs from a structured active-cell mask.

    With a target mask, faces adjacent to target cells are poloidal and all
    other domain edges are radial. This does not depend on their Cartesian
    orientation.
    """
    active = np.asarray(active, dtype=bool)
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if active.shape != r.shape or active.shape != z.shape:
        raise ValueError("active, R, and Z matrices must have identical shapes")
    has_target_mask = target is not None
    target = np.zeros_like(active) if target is None else np.asarray(target, bool)
    if target.shape != active.shape:
        raise ValueError("target mask must match active mask")

    found = []
    for axis in (0, 1):
        for side, sign in ((-1, -1.0), (1, 1.0)):
            neighbor_active = np.zeros_like(active)
            neighbor_target = np.zeros_like(active)
            src = [slice(None), slice(None)]
            dst = [slice(None), slice(None)]
            if side < 0:
                src[axis] = slice(0, -1)
                dst[axis] = slice(1, None)
            else:
                src[axis] = slice(1, None)
                dst[axis] = slice(0, -1)
            neighbor_active[tuple(dst)] = active[tuple(src)]
            neighbor_target[tuple(dst)] = target[tuple(src)]
            boundary = active & ~neighbor_active

            # Split target and non-target faces because they use different
            # velocity components. Outer array edges cannot be target faces.
            for is_target in (False, True):
                selection = boundary & (neighbor_target == is_target)
                # GENE-X target cells identify poloidal/parallel interfaces;
                # every other plasma-domain edge is a radial interface. This
                # classification is independent of its Cartesian orientation.
                kind = "poloidal" if (
                    is_target or (not has_target_mask and axis == 0)
                ) else "radial"
                varying = 1 - axis
                for run in _runs(np.argwhere(selection), varying):
                    centres = np.column_stack((r[tuple(run.T)], z[tuple(run.T)]))
                    found.append((kind, sign, run, centres, varying))

    components = []
    for number, (kind, sign, indices, centres, varying_axis) in enumerate(found):
        components.append(BoundaryComponent(
            f"{prefix}_{number:03d}", kind, indices, sign, centres,
            _structured_areas(indices, centres, r, z, varying_axis),
        ))
    return components


def eirene_boundary_components(gmtry, fnax_mask, fnay_mask):
    """Extract active EIRENE perimeter runs without assuming a named topology."""
    crx = np.asarray(gmtry["crx"])
    cry = np.asarray(gmtry["cry"])
    components = []
    for kind, mask, corners, axis, neighbor_keys in (
        ("poloidal", np.asarray(fnax_mask, bool), (0, 2), 0,
         ("leftix", "rightix")),
        ("radial", np.asarray(fnay_mask, bool), (2, 3), 1,
         ("bottomiy", "topiy")),
    ):
        r = np.mean(crx[..., list(corners)], axis=-1)
        z = np.mean(cry[..., list(corners)], axis=-1)
        # A fort.31 flux is present throughout the active plasma grid.  Its
        # boundary samples are the two outer transitions along its face axis;
        # inactive gaps naturally create additional target/cut components.
        for side_index, (side, sign) in enumerate(((-1, -1.0), (1, 1.0))):
            neighbor = np.zeros_like(mask)
            src = [slice(None), slice(None)]
            dst = [slice(None), slice(None)]
            if side < 0:
                src[axis] = slice(0, -1)
                dst[axis] = slice(1, None)
            else:
                src[axis] = slice(1, None)
                dst[axis] = slice(0, -1)
            neighbor[tuple(dst)] = mask[tuple(src)]
            topology_neighbor = gmtry.get(neighbor_keys[side_index])
            if topology_neighbor is None:
                boundary = mask & ~neighbor
            else:
                topology_neighbor = np.asarray(topology_neighbor)
                if topology_neighbor.size != mask.size:
                    raise ValueError(
                        f"SOLPS {neighbor_keys[side_index]} size does not "
                        "match the fort.31 mask"
                    )
                if topology_neighbor.shape != mask.shape:
                    topology_neighbor = topology_neighbor.reshape(
                        mask.shape, order="F"
                    )
                # Mask transitions find ordinary outer edges; negative SOLPS
                # neighbor indices additionally expose cuts which can be
                # adjacent in array storage but disconnected topologically.
                boundary = mask & ((topology_neighbor < 0) | ~neighbor)
            for run in _runs(np.argwhere(boundary), 1 - axis):
                centres = np.column_stack((r[tuple(run.T)], z[tuple(run.T)]))
                r0 = crx[tuple(run.T) + (np.full(len(run), corners[0]),)]
                r1 = crx[tuple(run.T) + (np.full(len(run), corners[1]),)]
                z0 = cry[tuple(run.T) + (np.full(len(run), corners[0]),)]
                z1 = cry[tuple(run.T) + (np.full(len(run), corners[1]),)]
                areas = np.pi * (r0 + r1) * np.hypot(r1 - r0, z1 - z0)
                components.append(BoundaryComponent(
                    f"eirene_{len(components):03d}", kind, run, sign, centres,
                    areas,
                ))
    components = _deduplicate_eirene_layers(components)
    components = _split_eirene_cut_segments(components, gmtry)
    components = [BoundaryComponent(
        f"eirene_{index:03d}", component.kind, component.indices,
        component.sign, component.centres, component.areas,
    ) for index, component in enumerate(components)]
    return _label_single_null_surfaces(components)


def _rename_component(component, name):
    return BoundaryComponent(
        name, component.kind, component.indices, component.sign,
        component.centres, component.areas,
    )


def _label_single_null_surfaces(components):
    """Attach physical names when the six-surface single-null pattern is clear.

    Unknown and more complicated topologies retain topology-independent
    ``eirene_NNN`` names rather than being assigned a potentially wrong label.
    """
    poloidal = [i for i, component in enumerate(components)
                if component.kind == "poloidal"]
    radial = [i for i, component in enumerate(components)
              if component.kind == "radial"]
    if len(poloidal) != 2 or len(radial) != 4:
        return components
    radial_signs = [components[i].sign for i in radial]
    unique, counts = np.unique(radial_signs, return_counts=True)
    if sorted(counts.tolist()) != [1, 3]:
        return components

    names = {}
    target_r = [float(np.mean(components[i].centres[:, 0])) for i in poloidal]
    names[poloidal[int(np.argmin(target_r))]] = "inner_target"
    names[poloidal[int(np.argmax(target_r))]] = "outer_target"

    sol_sign = float(unique[np.argmin(counts)])
    sol = [i for i in radial if components[i].sign == sol_sign]
    lower = [i for i in radial if components[i].sign != sol_sign]
    if len(sol) != 1 or len(lower) != 3:
        return components
    names[sol[0]] = "outer_SOL"
    # The cut splitter preserves the native poloidal ordering, with the core
    # boundary between the two private-flux-region legs.
    names[lower[1]] = "core"
    pfr = [lower[0], lower[2]]
    pfr_r = [float(np.mean(components[i].centres[:, 0])) for i in pfr]
    names[pfr[int(np.argmin(pfr_r))]] = "inner_PFR"
    names[pfr[int(np.argmax(pfr_r))]] = "outer_PFR"
    return [_rename_component(component, names[i])
            for i, component in enumerate(components)]


def _component_spacing(component):
    if len(component.centres) < 2:
        return np.inf
    values = np.linalg.norm(np.diff(component.centres, axis=0), axis=1)
    positive = values[values > 0]
    return float(np.median(positive)) if len(positive) else np.inf


def _hausdorff_distance(left, right):
    delta = left[:, None, :] - right[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    return float(max(np.max(np.min(distances, axis=1)),
                     np.max(np.min(distances, axis=0))))


def _deduplicate_eirene_layers(components):
    """Drop adjacent guard/plasma layers describing the same physical face."""
    kept = []
    for component in components:
        duplicate = False
        for prior in kept:
            if prior.kind != component.kind or prior.sign != component.sign:
                continue
            spacing = min(_component_spacing(prior),
                          _component_spacing(component))
            if np.isfinite(spacing) and _hausdorff_distance(
                prior.centres, component.centres
            ) < spacing:
                duplicate = True
                break
        if not duplicate:
            kept.append(component)
    return kept


def _split_eirene_cut_segments(components, gmtry):
    """Preserve SOLPS surface divisions along otherwise contiguous radial runs."""
    cuts = []
    for key in ("leftcut", "rightcut"):
        if key in gmtry:
            cuts.extend(int(value) for value in np.ravel(gmtry[key])
                        if int(value) >= 0)
    cuts = sorted(set(cuts))
    if not cuts:
        return components
    output = []
    for component in components:
        # SOLPS divertor cuts divide the lower/core/PFR side. The upper
        # outer-SOL boundary remains one physical surface across those cuts.
        if (component.kind != "radial" or component.sign > 0
                or len(component.indices) < 2):
            output.append(component)
            continue
        x = component.indices[:, 0]
        split_at = [position for position in range(1, len(x))
                    if any(x[position - 1] <= cut < x[position]
                           for cut in cuts)]
        for take in np.split(np.arange(len(x)), split_at):
            if len(take):
                output.append(BoundaryComponent(
                    component.name, component.kind, component.indices[take],
                    component.sign, component.centres[take],
                    component.areas[take],
                ))
    return output


def _minimum_distance(left, right):
    delta = left[:, None, :] - right[None, :, :]
    return float(np.min(np.linalg.norm(delta, axis=2)))


def match_component_groups(eirene, genex, *, distance_tolerance=None,
                           return_report=False):
    """Match GENE-X boundary fragments to EIRENE reference surfaces.

    GENE-X's rectangular matrix contains filler/NaN transitions which are not
    physical plasma boundaries. Such fragments are intentionally excluded when
    they are not close to an EIRENE boundary; they are not pairing failures.
    """
    if not eirene or not genex:
        raise ValueError("both meshes must provide boundary components")
    spans = [np.ptp(c.centres, axis=0).max() for c in (*eirene, *genex)]
    scale = max(spans, default=0.0)
    tolerance = distance_tolerance
    if tolerance is None:
        spacings = []
        for component in (*eirene, *genex):
            if len(component.centres) > 1:
                spacings.extend(np.linalg.norm(
                    np.diff(component.centres, axis=0), axis=1
                ))
        tolerance = 3 * np.median(spacings) if spacings else max(scale * .05, 1e-9)

    nodes = [("e", i) for i in range(len(eirene))] + [
        ("g", i) for i in range(len(genex))
    ]
    adjacency = {node: set() for node in nodes}
    for i, ec in enumerate(eirene):
        candidates = [(j, _minimum_distance(ec.centres, gc.centres))
                      for j, gc in enumerate(genex) if gc.kind == ec.kind]
        if not candidates:
            raise ValueError(f"no GENE-X {ec.kind} component for {ec.name}")
        close = [(j, distance) for j, distance in candidates
                 if distance <= tolerance]
        if not close:
            j, distance = min(candidates, key=lambda item: item[1])
            if distance > max(10 * tolerance, .1 * max(scale, 1e-9)):
                raise ValueError(f"cannot geometrically pair {ec.name}")
            close = [(j, distance)]
        for j, _ in close:
            adjacency[("e", i)].add(("g", j))
            adjacency[("g", j)].add(("e", i))

    excluded = []
    for j, component in enumerate(genex):
        if adjacency[("g", j)]:
            continue
        candidates = [
            _minimum_distance(component.centres, ec.centres)
            for ec in eirene if ec.kind == component.kind
        ]
        excluded.append({
            "name": component.name,
            "kind": component.kind,
            "indices": component.indices.tolist(),
            "centre": np.mean(component.centres, axis=0).tolist(),
            "nearest_eirene_distance": min(candidates) if candidates else None,
            "reason": "not close to the EIRENE plasma-domain boundary",
        })

    groups, seen = [], set()
    for node in nodes:
        if not adjacency[node]:
            seen.add(node)
            continue
        if node in seen:
            continue
        stack, group = [node], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            group.append(current)
            stack.extend(adjacency[current] - seen)
        eids = [i for side, i in group if side == "e"]
        gids = [i for side, i in group if side == "g"]
        groups.append((eids, gids))
    report = {
        "distance_tolerance": float(tolerance),
        "eirene_component_count": len(eirene),
        "genex_candidate_component_count": len(genex),
        "genex_matched_component_count": len(genex) - len(excluded),
        "genex_excluded_component_count": len(excluded),
        "rejected_component_count": 0,
        "excluded_genex_components": excluded,
        "rejected_components": [],
    }
    return (groups, report) if return_report else groups


def match_boundary_faces(eirene, genex, *, distance_tolerance=None):
    """Assign individual GENE-X faces to EIRENE surface groups.

    Components are useful for identifying continuity, but a whole component is
    not accepted merely because one endpoint is close. Every retained GENE-X
    face independently passes the geometric-distance test.
    """
    if not eirene or not genex:
        raise ValueError("both meshes must provide boundary components")
    if distance_tolerance is None:
        spacings = []
        for component in (*eirene, *genex):
            if len(component.centres) > 1:
                spacings.extend(np.linalg.norm(
                    np.diff(component.centres, axis=0), axis=1
                ))
        distance_tolerance = (
            3 * np.median(spacings) if spacings else 1e-9
        )

    assignments = {}
    nearest_distance = {}
    e_to_g = {i: set() for i in range(len(eirene))}
    fallback_matches = []
    for j, gc in enumerate(genex):
        compatible = [(i, ec) for i, ec in enumerate(eirene)
                      if ec.kind == gc.kind]
        for face_index, centre in enumerate(gc.centres):
            choices = [
                (i, float(np.min(np.linalg.norm(ec.centres - centre, axis=1))))
                for i, ec in compatible
            ]
            if not choices:
                nearest_distance[(j, face_index)] = None
                continue
            e_index, distance = min(choices, key=lambda item: item[1])
            nearest_distance[(j, face_index)] = distance
            if distance <= distance_tolerance:
                assignments[(j, face_index)] = e_index
                e_to_g[e_index].add(j)

    spans = [np.ptp(component.centres, axis=0).max()
             for component in (*eirene, *genex)]
    fallback_limit = max(10 * distance_tolerance,
                         0.1 * max(spans, default=0.0))
    for e_index, ec in enumerate(eirene):
        if e_to_g[e_index]:
            continue
        candidates = [
            (j, face_index, centre)
            for j, gc in enumerate(genex) if gc.kind == ec.kind
            for face_index, centre in enumerate(gc.centres)
        ]
        for e_face, centre in enumerate(ec.centres):
            available = [item for item in candidates
                         if (item[0], item[1]) not in assignments]
            if not available:
                break
            j, face_index, g_centre = min(
                available,
                key=lambda item: np.linalg.norm(item[2] - centre),
            )
            distance = float(np.linalg.norm(g_centre - centre))
            if distance > fallback_limit:
                continue
            assignments[(j, face_index)] = e_index
            nearest_distance[(j, face_index)] = distance
            e_to_g[e_index].add(j)
            fallback_matches.append({
                "eirene_component": ec.name,
                "eirene_face_index": e_face,
                "genex_component": genex[j].name,
                "genex_face_index": face_index,
                "distance": distance,
            })

    uncovered = [eirene[i].name for i, linked in e_to_g.items() if not linked]
    if uncovered:
        raise ValueError(
            "EIRENE boundary components have no nearby native GENE-X faces: "
            f"{uncovered}; tolerance={distance_tolerance:.6e} m"
        )

    # EIRENE defines the authoritative surface divisions (including SOLPS
    # cuts). A GENE-X run may cross several such surfaces and is split below,
    # rather than merging those EIRENE surfaces back together.
    groups_e = [[index] for index in range(len(eirene))]

    selected = []
    groups = []
    for group_index, eids in enumerate(groups_e):
        gids = []
        for j, gc in enumerate(genex):
            # Split by the assigned EIRENE component so every native face uses
            # the flux-coordinate outward sign of its physical surface. The
            # Cartesian side of a stair-stepped GENE-X mask is not a radial or
            # poloidal sign convention.
            for e_index in eids:
                face_indices = [
                    face_index for face_index in range(len(gc.indices))
                    if assignments.get((j, face_index)) == e_index
                ]
                if not face_indices:
                    continue
                take = np.asarray(face_indices, dtype=int)
                gids.append(len(selected))
                selected.append(BoundaryComponent(
                    f"genex_group_{group_index:03d}_{gc.name}_to_"
                    f"{eirene[e_index].name}",
                    gc.kind,
                    gc.indices[take],
                    eirene[e_index].sign,
                    gc.centres[take],
                    gc.areas[take],
                ))
        groups.append((eids, gids))

    excluded = []
    excluded_face_count = 0
    for j, gc in enumerate(genex):
        face_indices = [i for i in range(len(gc.indices))
                        if (j, i) not in assignments]
        if not face_indices:
            continue
        excluded_face_count += len(face_indices)
        distances = [nearest_distance[(j, i)] for i in face_indices]
        excluded.append({
            "source_component": gc.name,
            "kind": gc.kind,
            "indices": gc.indices[face_indices].tolist(),
            "centres": gc.centres[face_indices].tolist(),
            "nearest_eirene_distances": distances,
            "reason": "outside EIRENE boundary distance tolerance",
        })
    matched_face_count = sum(len(component.indices) for component in selected)
    report = {
        "distance_tolerance": float(distance_tolerance),
        "fallback_distance_limit": float(fallback_limit),
        "eirene_component_count": len(eirene),
        "genex_candidate_component_count": len(genex),
        "genex_candidate_face_count": matched_face_count + excluded_face_count,
        "genex_matched_component_count": len(selected),
        "genex_matched_face_count": matched_face_count,
        "genex_excluded_component_count": len(excluded),
        "genex_excluded_face_count": excluded_face_count,
        "rejected_component_count": 0,
        "rejected_face_count": 0,
        "excluded_genex_components": excluded,
        "rejected_components": [],
        "fallback_match_count": len(fallback_matches),
        "fallback_matches": fallback_matches,
    }
    return groups, selected, report


def map_boundary_cell_sequences(eirene, genex, *, distance_tolerance=None):
    """Map each EIRENE surface to an ordered sequence of native GENE-X cells.

    EIRENE supplies the physical surface ordering. Nearest native boundary
    cells are deduplicated in that order and integrated as an axisymmetric
    centre line. This avoids summing the Cartesian sides of a stair-stepped
    mask, whose side lengths are not the area measure for GENE-X ``u_rad``.
    """
    if not eirene or not genex:
        raise ValueError("both meshes must provide boundary components")
    spacings = []
    for component in (*eirene, *genex):
        if len(component.centres) > 1:
            spacings.extend(np.linalg.norm(
                np.diff(component.centres, axis=0), axis=1
            ))
    if distance_tolerance is None:
        distance_tolerance = 3 * np.median(spacings) if spacings else 1e-9
    spans = [np.ptp(component.centres, axis=0).max()
             for component in (*eirene, *genex)]
    fallback_limit = max(10 * distance_tolerance,
                         0.1 * max(spans, default=0.0))

    selected = []
    groups = []
    used_faces = set()
    fallback_matches = []
    for e_index, ec in enumerate(eirene):
        candidates = [
            (j, face_index, tuple(int(v) for v in gc.indices[face_index]),
             gc.centres[face_index], gc.areas[face_index])
            for j, gc in enumerate(genex) if gc.kind == ec.kind
            for face_index in range(len(gc.indices))
        ]
        if not candidates:
            raise ValueError(f"no native GENE-X {ec.kind} faces for {ec.name}")
        ordered = []
        seen_cells = set()
        for e_face, centre in enumerate(ec.centres):
            choice = min(candidates,
                         key=lambda item: np.linalg.norm(item[3] - centre))
            j, face_index, cell_index, g_centre, source_area = choice
            distance = float(np.linalg.norm(g_centre - centre))
            if distance > fallback_limit:
                raise ValueError(
                    f"{ec.name} face {e_face} is {distance:.6e} m from the "
                    f"nearest GENE-X boundary (limit {fallback_limit:.6e} m)"
                )
            if distance > distance_tolerance:
                fallback_matches.append({
                    "eirene_component": ec.name,
                    "eirene_face_index": e_face,
                    "genex_component": genex[j].name,
                    "genex_face_index": face_index,
                    "distance": distance,
                })
            used_faces.add((j, face_index))
            if cell_index in seen_cells:
                continue
            seen_cells.add(cell_index)
            ordered.append((cell_index, np.asarray(g_centre), source_area))
        if not ordered:
            raise ValueError(f"{ec.name} mapped to no unique GENE-X cells")
        indices = np.asarray([item[0] for item in ordered], dtype=int)
        centres = np.asarray([item[1] for item in ordered], dtype=float)
        areas = (axisymmetric_point_areas(centres) if len(centres) > 1
                 else np.asarray([ordered[0][2]], dtype=float))
        groups.append(([e_index], [len(selected)]))
        selected.append(BoundaryComponent(
            f"genex_surface_for_{ec.name}", ec.kind, indices, ec.sign,
            centres, areas,
        ))

    excluded = []
    excluded_face_count = 0
    for j, gc in enumerate(genex):
        take = [i for i in range(len(gc.indices)) if (j, i) not in used_faces]
        if not take:
            continue
        excluded_face_count += len(take)
        excluded.append({
            "source_component": gc.name,
            "kind": gc.kind,
            "indices": gc.indices[take].tolist(),
            "centres": gc.centres[take].tolist(),
            "reason": "not selected by an EIRENE surface sequence",
        })
    matched_face_count = sum(len(component.indices) for component in selected)
    report = {
        "distance_tolerance": float(distance_tolerance),
        "fallback_distance_limit": float(fallback_limit),
        "eirene_component_count": len(eirene),
        "genex_candidate_component_count": len(genex),
        "genex_candidate_face_count": sum(len(c.indices) for c in genex),
        "genex_matched_component_count": len(selected),
        "genex_matched_face_count": matched_face_count,
        "genex_excluded_component_count": len(excluded),
        "genex_excluded_face_count": excluded_face_count,
        "rejected_component_count": 0,
        "rejected_face_count": 0,
        "excluded_genex_components": excluded,
        "rejected_components": [],
        "fallback_match_count": len(fallback_matches),
        "fallback_matches": fallback_matches,
    }
    return groups, selected, report


def _species_view(array, species_index):
    values = np.asarray(array)
    return values if values.ndim == 2 else values[..., species_index]


def directional_rates(values, components):
    """Return outward, inward-magnitude, and net rates."""
    outward = 0.0
    inward = 0.0
    for component in components:
        local = np.asarray(values[tuple(component.indices.T)], dtype=float)
        signed = component.sign * local
        outward += np.sum(np.maximum(signed, 0.0) * component.areas)
        inward += np.sum(np.maximum(-signed, 0.0) * component.areas)
    return float(outward), float(inward), float(outward - inward)


def outgoing_rate(values, components):
    return directional_rates(values, components)[0]


def normalize_eirene_particle_fluxes(
    fort31, eirene_components, genex_fluxes, genex_components, groups,
    species_names, *, factor_min=.5, factor_max=2.0,
):
    """Conserve outgoing rate for each species and matched surface group."""
    records = []
    for species_index, species in enumerate(species_names):
        native = {
            kind: _species_view(values, species_index)
            for kind, values in genex_fluxes.items()
        }
        for group_index, (eids, gids) in enumerate(groups):
            ecs = [eirene_components[i] for i in eids]
            gcs = [genex_components[i] for i in gids]
            e_rates = np.sum([directional_rates(
                _species_view(fort31["fnax" if c.kind == "poloidal" else "fnay"],
                              species_index), [c]
            ) for c in ecs], axis=0)
            g_rates = np.sum([
                directional_rates(native[c.kind], [c]) for c in gcs
            ], axis=0)
            e_total, e_inward, e_net = (float(value) for value in e_rates)
            g_total, g_inward, g_net = (float(value) for value in g_rates)
            gross = max(e_total + e_inward, g_total + g_inward)
            net_residual = (abs(g_net - e_net) / gross if gross > 0.0
                            else 0.0)
            record = {"surface_group": group_index, "species": species,
                      "eirene_components": [c.name for c in ecs],
                      "genex_components": [c.name for c in gcs],
                      # Keep the original names for diagnostics consumers.
                      "eirene_before": e_total, "genex": g_total,
                      "eirene_outward_before": e_total,
                      "genex_outward": g_total,
                      "eirene_inward": e_inward,
                      "genex_inward": g_inward,
                      "eirene_net_before": e_net,
                      "genex_net": g_net,
                      "net_residual": net_residual}
            if not all(np.isfinite(value) for value in
                       (*e_rates, *g_rates, net_residual)):
                raise ValueError(f"non-finite boundary total: {record}")
            if e_total == 0.0:
                all_zero = all(np.all(_species_view(
                    fort31["fnax" if c.kind == "poloidal" else "fnay"],
                    species_index,
                )[tuple(c.indices.T)] == 0) for c in ecs)
                if g_total != 0.0 and all_zero:
                    raise ValueError(f"EIRENE boundary is entirely zero: {record}")
                record.update(factor=1.0, eirene_after=e_total,
                              eirene_outward_after=e_total,
                              eirene_net_after=e_net,
                              warning="no outgoing EIRENE flux; component skipped")
                records.append(record)
                continue
            factor = g_total / e_total
            if not np.isfinite(factor) or not factor_min <= factor <= factor_max:
                raise ValueError(
                    f"boundary factor {factor:.6g} outside "
                    f"[{factor_min}, {factor_max}]: {record}"
                )
            for component in ecs:
                key = "fnax" if component.kind == "poloidal" else "fnay"
                view = _species_view(fort31[key], species_index)
                index = tuple(component.indices.T)
                local = view[index]
                outward = component.sign * local > 0
                local[outward] *= factor
                view[index] = local
            after_rates = np.sum([directional_rates(
                _species_view(fort31["fnax" if c.kind == "poloidal" else "fnay"],
                              species_index), [c]
            ) for c in ecs], axis=0)
            after, _, net_after = (float(value) for value in after_rates)
            record.update(factor=factor, eirene_after=after,
                          eirene_outward_after=after,
                          eirene_net_after=net_after)
            records.append(record)
    for species in species_names:
        species_records = [item for item in records
                           if item["species"] == species]
        denominator = sum(max(
            item["genex_outward"] + item["genex_inward"],
            item["eirene_outward_before"] + item["eirene_inward"],
        ) for item in species_records)
        differences = [abs(item["genex_net"] - item["eirene_net_before"])
                       for item in species_records]
        global_residual = (sum(differences) / denominator
                           if denominator > 0.0 else 0.0)
        for item, difference in zip(species_records, differences):
            item["global_net_residual"] = global_residual
            item["global_net_residual_contribution"] = (
                difference / denominator if denominator > 0.0 else 0.0
            )
    return records


def write_normalization_diagnostics(path, records):
    Path(path).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
