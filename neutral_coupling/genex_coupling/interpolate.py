from scipy.interpolate import (griddata, LinearNDInterpolator,
                               NearestNDInterpolator)
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import warnings


@dataclass(frozen=True)
class RegionalDiagnostic:
    global_residual: float
    normalized_regional_error: float
    worst_relative_error: float
    worst_bounds: tuple | None
    worst_source_fraction: float
    worst_component: str | None
    significant: bool


class ConservativeSourceMapper:
    """Cached linear interpolation followed by conservative normalization.

    Source values are interpreted as EIRENE cell averages and ``source_weights``
    as triangle volumes.  Destination weights are the GENE-X discrete source
    integration weights (``jacobian * RZw`` for one toroidal plane).
    """

    def __init__(self, source_points, destination_points, source_weights,
                 destination_weights, *, levels=2,
                 minimum_source_fraction=0.01,
                 regional_error_threshold=0.10):
        self.source_points = np.asarray(source_points, dtype=float)
        self.destination_points = np.asarray(destination_points, dtype=float)
        self.source_weights = self._weights(source_weights, len(self.source_points),
                                            "EIRENE")
        self.destination_weights = self._weights(
            destination_weights, len(self.destination_points), "GENE-X"
        )
        self.minimum_source_fraction = float(minimum_source_fraction)
        self.regional_error_threshold = float(regional_error_threshold)
        self.operator = self._linear_operator()
        self.regions = self._regional_masks(levels)

    @staticmethod
    def _weights(values, size, label):
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size != size:
            raise ValueError(f"{label} weights have {values.size} values, expected {size}")
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"{label} integration weights must be finite and positive")
        return values

    def _linear_operator(self):
        tri = Delaunay(self.source_points)
        simplex = tri.find_simplex(self.destination_points)
        inside = simplex >= 0
        rows = np.repeat(np.flatnonzero(inside), self.source_points.shape[1] + 1)
        selected = simplex[inside]
        transform = tri.transform[selected]
        delta = self.destination_points[inside] - transform[:, -1]
        bary = np.einsum("nij,nj->ni", transform[:, :-1], delta)
        bary = np.column_stack((bary, 1.0 - bary.sum(axis=1)))
        cols = tri.simplices[selected].reshape(-1)
        return csr_matrix(
            (bary.reshape(-1), (rows, cols)),
            shape=(len(self.destination_points), len(self.source_points)),
        )

    def _regional_masks(self, levels):
        points = np.vstack((self.source_points, self.destination_points))
        r_edges = (float(points[:, 0].min()), float(points[:, 0].max()))
        z_edges = (float(points[:, 1].min()), float(points[:, 1].max()))
        regions = []
        for level in range(1, int(levels) + 1):
            count = 2 ** level
            rs = np.linspace(*r_edges, count + 1)
            zs = np.linspace(*z_edges, count + 1)
            for i in range(count):
                for j in range(count):
                    # Close the outermost bins so every centre is included once.
                    src = self._box_mask(self.source_points, rs, zs, i, j, count)
                    dst = self._box_mask(self.destination_points, rs, zs, i, j, count)
                    regions.append(((rs[i], rs[i + 1], zs[j], zs[j + 1]), src, dst))
        return regions

    @staticmethod
    def _box_mask(points, rs, zs, i, j, count):
        r_upper = points[:, 0] <= rs[i + 1] if i == count - 1 else points[:, 0] < rs[i + 1]
        z_upper = points[:, 1] <= zs[j + 1] if j == count - 1 else points[:, 1] < zs[j + 1]
        return ((points[:, 0] >= rs[i]) & r_upper &
                (points[:, 1] >= zs[j]) & z_upper)

    def map(self, source, *, label="source"):
        source = np.asarray(source, dtype=float).reshape(-1)
        if source.size != self.source_weights.size:
            raise ValueError(
                f"{label} has {source.size} values, expected {self.source_weights.size}"
            )
        corrected = np.zeros(len(self.destination_points), dtype=float)
        for sign, selector in (("positive", source > 0), ("negative", source < 0)):
            source_part = np.where(selector, source, 0.0)
            native = float(np.dot(source_part, self.source_weights))
            mapped_part = np.asarray(self.operator @ source_part).reshape(-1)
            destination = float(np.dot(mapped_part, self.destination_weights))
            if native == 0.0:
                continue
            tolerance = np.finfo(float).eps * abs(native) * 32
            if abs(destination) <= tolerance:
                raise ValueError(
                    f"Cannot conservatively map {label} {sign} contribution: "
                    "the EIRENE volume integral is nonzero but linear interpolation "
                    "has no support on active GENE-X cells"
                )
            corrected += mapped_part * (native / destination)
        return corrected, self.diagnose(source, corrected)

    def diagnose(self, source, destination):
        source = np.asarray(source).reshape(-1)
        destination = np.asarray(destination).reshape(-1)
        absolute_total = float(np.dot(np.abs(source), self.source_weights))
        scale = max(absolute_total, np.finfo(float).tiny)
        residuals = []
        for source_part, destination_part in (
            (source, destination),
            (np.maximum(source, 0), np.maximum(destination, 0)),
            (np.minimum(source, 0), np.minimum(destination, 0)),
        ):
            native = float(np.dot(source_part, self.source_weights))
            mapped = float(np.dot(destination_part, self.destination_weights))
            residuals.append(abs(mapped - native) / scale)
        errors = []
        components = (
            ("net", source, destination),
            ("positive", np.maximum(source, 0), np.maximum(destination, 0)),
            ("negative", np.minimum(source, 0), np.minimum(destination, 0)),
        )
        for component, source_component, destination_component in components:
            for bounds, src_mask, dst_mask in self.regions:
                src_total = float(np.dot(source_component[src_mask],
                                         self.source_weights[src_mask]))
                dst_total = float(np.dot(destination_component[dst_mask],
                                         self.destination_weights[dst_mask]))
                fraction = abs(src_total) / scale
                error = abs(dst_total - src_total)
                relative = error / max(abs(src_total),
                                       self.minimum_source_fraction * scale)
                if fraction >= self.minimum_source_fraction:
                    errors.append((relative, error, bounds, fraction, component))
        worst = max(errors, default=(0.0, 0.0, None, 0.0, None),
                    key=lambda item: item[0])
        regional_error = sum(item[1] for item in errors) / scale
        return RegionalDiagnostic(
            global_residual=max(residuals),
            normalized_regional_error=regional_error,
            worst_relative_error=worst[0], worst_bounds=worst[2],
            worst_source_fraction=worst[3],
            worst_component=worst[4],
            significant=worst[0] > self.regional_error_threshold,
        )

def interpolate_all_sources(tria, source_dict, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0, mapper=None,
                            diagnostics=None):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for mom, mom_block in source_dict.items():
        for species, species_block in mom_block.items():
            for source, value in species_block.items():
                label = f"{mom}/{species}/{source}"
                if mapper is None:
                    result = interpolate_source(tria, value, grid_r, grid_z,
                                method=method, fill_mode=fill_mode,
                                fill_value=fill_value)
                else:
                    result, report = mapper.map(value, label=label)
                    if diagnostics is not None:
                        diagnostics[label] = report
                out[mom][species][source] = result.reshape(1, len(grid_r))
    return out

# TODO add better out of bounds value/check
def interpolate_source(tria, source, grid_r, grid_z, method, fill_mode,
                       fill_value):
    interp_source = griddata(tria.incenter, source, (grid_r, grid_z),
                                        method=method)
    # Create shapely.MultiPolygon
    # Check if points of grid_r, grid_z are contained within polygon
    # 0 out points outside of domain or something else special
    # This is important for something like the hl2a baffling
    if (fill_mode == "constant"):
        interp_source[np.isnan(interp_source)] = fill_value
    elif (fill_mode == "nearest"):
        pass

    return interp_source

def build_triangulation(r, z):
    points = np.column_stack([r.ravel(), z.ravel()])
    return Delaunay(points)

# There are 3 interpolation routines in torx
# TODO add better out of bounds value/check
def interp_moments(gmtry, tri, field, ind):
    VALID_INDS = {
        (0, 2),
        (2, 3),
        (0, 1, 2, 3),
    }
    if isinstance(ind, slice):
        ind_tuple = tuple(range(*ind.indices(4)))
    else:
        ind_tuple = tuple(ind)
    if ind_tuple not in VALID_INDS:
        warnings.warn(
            f"interp_moments received non-standard ind={ind_tuple}. "
            "This is geometrically valid but not used in standard physics.",
            UserWarning,
            stacklevel=2,
        )
    r = np.mean(gmtry["crx"][:,:,ind],2)
    z = np.mean(gmtry["cry"][:,:,ind],2)
    values = np.asarray(field).ravel()
    if values.size != tri.points.shape[0]:
        raise ValueError(
            "GENE-X field size does not match the interpolation triangulation"
        )
    # Linear interpolation is preferred within the GENE-X plasma domain.  An
    # EIRENE face can lie just beyond that domain (notably at a target), where
    # extrapolating linearly is undefined; use its nearest plasma cell there.
    interp = LinearNDInterpolator(tri, values)
    xi = np.column_stack([r.ravel(), z.ravel()])
    interp_data = np.asarray(interp(xi))
    outside = tri.find_simplex(xi) < 0
    if np.any(outside):
        nearest = NearestNDInterpolator(tri.points, values)
        interp_data[outside] = nearest(xi[outside])
    interp_data = interp_data.reshape(r.shape)
    # Preserve the prior fail-safe for unusable samples within the domain.
    interp_data[np.isnan(interp_data)] = 0
    return interp_data
