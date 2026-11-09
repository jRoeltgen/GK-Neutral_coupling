import json

import numpy as np
import pytest

from neutral_coupling.genex_coupling.boundary_flux import (
    BoundaryComponent, _label_single_null_surfaces, axisymmetric_point_areas,
    eirene_boundary_components,
    map_boundary_cell_sequences, match_boundary_faces, match_component_groups,
    normalize_eirene_particle_fluxes, structured_boundary_components,
    write_normalization_diagnostics,
)


def component(name, kind, indices, sign, centres, areas=None):
    centres = np.asarray(centres, dtype=float)
    return BoundaryComponent(
        name, kind, np.asarray(indices, dtype=int), sign, centres,
        np.ones(len(centres)) if areas is None else np.asarray(areas, float),
    )


def test_axisymmetric_areas_include_native_endpoint_half_cells():
    centres = np.array([[2.0, 0.0], [2.0, 0.5], [2.0, 1.0]])
    np.testing.assert_allclose(
        axisymmetric_point_areas(centres), np.full(3, 2 * np.pi)
    )


def test_structured_components_discover_four_disconnected_targets():
    active = np.ones((7, 8), dtype=bool)
    target = np.zeros_like(active)
    target[1, 1:3] = True
    target[1, 5:7] = True
    target[5, 1:3] = True
    target[5, 5:7] = True
    active[target] = False
    rr, zz = np.meshgrid(np.arange(8.0), np.arange(7.0))
    components = structured_boundary_components(
        active, rr, zz, target=target, prefix="synthetic_dn"
    )
    target_components = [c for c in components if c.kind == "poloidal"]
    assert len(target_components) >= 4
    assert all(c.name.startswith("synthetic_dn_") for c in components)


def test_eirene_neighbor_topology_discovers_four_cut_surfaces():
    shape = (7, 8)
    crx = np.zeros(shape + (4,))
    cry = np.zeros_like(crx)
    for i in range(shape[0]):
        for j in range(shape[1]):
            crx[i, j] = [j + 1, j + 2, j + 1, j + 2]
            cry[i, j] = [i, i, i + 1, i + 1]
    leftix = np.zeros(shape, dtype=int)
    for i in (1, 5):
        leftix[i, 1:3] = -1
        leftix[i, 5:7] = -1
    gmtry = {
        "crx": crx, "cry": cry,
        "leftix": leftix, "rightix": np.zeros(shape, dtype=int),
        "bottomiy": np.zeros(shape, dtype=int),
        "topiy": np.zeros(shape, dtype=int),
    }
    components = eirene_boundary_components(
        gmtry, np.ones(shape, bool), np.ones(shape, bool)
    )
    # Four cut surfaces plus any ordinary outer transitions of the mask.
    assert len([c for c in components if c.kind == "poloidal"]) >= 4


def test_single_null_surfaces_receive_physical_labels():
    surfaces = [
        component("e0", "poloidal", [[0, 0]], -1, [[0.6, -0.4]]),
        component("e1", "poloidal", [[1, 0]], 1, [[0.9, -0.7]]),
        component("e2", "radial", [[0, 0]], -1, [[0.6, -0.5]]),
        component("e3", "radial", [[1, 0]], -1, [[0.9, 0.0]]),
        component("e4", "radial", [[2, 0]], -1, [[0.7, -0.7]]),
        component("e5", "radial", [[3, 0]], 1, [[0.8, 0.4]]),
    ]
    assert [item.name for item in _label_single_null_surfaces(surfaces)] == [
        "inner_target", "outer_target", "inner_PFR", "core", "outer_PFR",
        "outer_SOL",
    ]


def test_many_to_many_geometric_matching_groups_fragments():
    eirene = [
        component("e0", "radial", [[0, 0]], 1, [[1.0, 0.0]]),
        component("e1", "radial", [[0, 1]], 1, [[1.1, 0.0]]),
    ]
    genex = [component(
        "g0", "radial", [[0, 0], [0, 1]], 1,
        [[1.0, 0.01], [1.1, 0.01]],
    )]
    groups = match_component_groups(eirene, genex, distance_tolerance=.05)
    assert groups == [([0, 1], [0])]


def test_distant_genex_matrix_edge_is_reported_as_excluded():
    eirene = [component("e0", "radial", [[0, 0]], 1, [[1.0, 0.0]])]
    genex = [
        component("g0", "radial", [[0, 0]], 1, [[1.01, 0.0]]),
        component("filler_edge", "radial", [[8, 8]], 1, [[9.0, 9.0]]),
    ]
    groups, report = match_component_groups(
        eirene, genex, distance_tolerance=.05, return_report=True
    )
    assert groups == [([0], [0])]
    assert report["genex_matched_component_count"] == 1
    assert report["genex_excluded_component_count"] == 1
    assert report["excluded_genex_components"][0]["name"] == "filler_edge"


def test_face_matching_does_not_accept_entire_run_from_one_close_endpoint():
    eirene = [component("e", "radial", [[0, 0]], 1, [[1.0, 0.0]])]
    genex = [component(
        "long_run", "radial", [[0, 0], [0, 1], [0, 2]], 1,
        [[1.01, 0.0], [2.0, 0.0], [3.0, 0.0]], areas=[1, 2, 3],
    )]
    groups, selected, report = match_boundary_faces(
        eirene, genex, distance_tolerance=.05
    )
    assert groups == [([0], [0])]
    np.testing.assert_array_equal(selected[0].indices, [[0, 0]])
    np.testing.assert_array_equal(selected[0].areas, [1])
    assert report["genex_matched_face_count"] == 1
    assert report["genex_excluded_face_count"] == 2


def test_matched_native_face_uses_eirene_flux_coordinate_sign():
    eirene = [component("outer", "radial", [[0, 0]], 1, [[1.0, 0.0]])]
    # Cartesian matrix-side sign is deliberately opposite.
    genex = [component("stair_step", "radial", [[2, 3]], -1,
                       [[1.01, 0.0]])]
    _groups, selected, _report = match_boundary_faces(
        eirene, genex, distance_tolerance=.05
    )
    assert selected[0].sign == 1


def test_surface_sequence_deduplicates_cells_and_uses_curve_length():
    eirene = [component(
        "outer_sol", "radial", [[0, 0], [0, 1], [0, 2]], 1,
        [[2.0, 0.0], [2.0, 0.5], [2.0, 1.0]],
    )]
    # The same native cell can appear on two sides of a stair-step boundary.
    genex = [component(
        "stair_step", "radial", [[3, 4], [3, 4], [3, 5]], -1,
        [[2.0, 0.0], [2.0, 0.01], [2.0, 1.0]], areas=[.01, .01, .01],
    )]
    groups, selected, report = map_boundary_cell_sequences(
        eirene, genex, distance_tolerance=.6
    )
    assert groups == [([0], [0])]
    np.testing.assert_array_equal(selected[0].indices, [[3, 4], [3, 5]])
    np.testing.assert_allclose(selected[0].areas, [4 * np.pi, 4 * np.pi])
    assert selected[0].sign == 1
    assert report["genex_matched_face_count"] == 2


def test_surface_sequence_can_override_boundary_distance_error():
    eirene = [component(
        "eirene", "radial", [[0, 0]], 1, [[0.0, 0.0]],
    )]
    genex = [component(
        "genex", "radial", [[0, 0]], 1, [[10.0, 0.0]],
    )]

    with pytest.raises(ValueError, match="nearest GENE-X boundary"):
        map_boundary_cell_sequences(
            eirene, genex, distance_tolerance=0.1
        )

    groups, selected, report = map_boundary_cell_sequences(
        eirene, genex, distance_tolerance=0.1,
        allow_boundary_misalignment=True,
    )

    assert groups == [([0], [0])]
    assert len(selected) == 1
    assert report["allow_boundary_misalignment"] is True
    assert report["fallback_match_count"] == 1
    assert report["fallback_matches"][0]["distance"] == pytest.approx(10.0)


def test_normalization_scales_only_outgoing_values_per_species():
    fort31 = {
        "fnax": np.array([[[2.0, 4.0], [-3.0, -6.0]]]),
        "fnay": np.zeros((1, 2, 2)),
    }
    eirene = [component(
        "e", "poloidal", [[0, 0], [0, 1]], 1,
        [[1.0, 0.0], [1.0, 1.0]],
    )]
    genex = [component(
        "g", "poloidal", [[0, 0], [0, 1]], 1,
        [[1.0, 0.0], [1.0, 1.0]],
    )]
    native = {
        "poloidal": np.array([[[3.0, 6.0], [-99.0, -99.0]]]),
        "radial": np.zeros((1, 2, 2)),
    }
    records = normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D", "T"],
    )
    np.testing.assert_allclose(fort31["fnax"][0, 0], [3.0, 6.0])
    np.testing.assert_allclose(fort31["fnax"][0, 1], [-3.0, -6.0])
    assert [r["factor"] for r in records] == [1.5, 1.5]
    assert records[0]["eirene_inward"] == 3.0
    assert records[0]["genex_inward"] == 99.0
    assert records[0]["eirene_net_before"] == -1.0
    assert records[0]["genex_net"] == -96.0
    assert records[0]["net_residual"] == pytest.approx(95 / 102)
    assert records[0]["eirene_net_after"] == 0.0
    assert records[0]["global_net_residual"] == pytest.approx(95 / 102)
    assert records[0]["global_net_residual_contribution"] == pytest.approx(
        95 / 102
    )


def test_radial_scaling_multiplies_normalization_factor():
    fort31 = {"fnax": np.zeros((1, 2)),
              "fnay": np.array([[2.0, -4.0]])}
    eirene = [component("e", "radial", [[0, 0], [0, 1]], 1,
                        [[1, 0], [1, 1]])]
    genex = [component("g", "radial", [[0, 0], [0, 1]], 1,
                       [[1, 0], [1, 1]])]
    native = {"poloidal": np.zeros((1, 1)),
              "radial": np.array([[3.0, -4.0]])}

    records = normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D"],
        radial_flux_scaling=2.0,
    )

    # Normalization contributes 3/2 and radial scaling contributes 2.
    np.testing.assert_allclose(fort31["fnay"], [[6.0, -8.0]])
    assert records[0]["factor"] == pytest.approx(1.5)
    assert records[0]["radial_flux_scaling"] == 2.0


def test_radial_scaling_applies_when_no_outgoing_flux():
    fort31 = {"fnax": np.zeros((1, 1)), "fnay": np.array([[-4.0]])}
    eirene = [component("e", "radial", [[0, 0]], 1, [[1, 0]])]
    genex = [component("g", "radial", [[0, 0]], 1, [[1, 0]])]
    native = {"poloidal": np.zeros((1, 1)),
              "radial": np.array([[-3.0]])}

    records = normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D"],
        radial_flux_scaling=0.25,
    )

    assert fort31["fnay"][0, 0] == pytest.approx(-1.0)
    assert records[0]["factor"] == 1.0
    assert "normalization skipped" in records[0]["warning"]


def test_radial_scaling_does_not_change_poloidal_flux():
    fort31 = {"fnax": np.array([[2.0]]), "fnay": np.zeros((1, 1))}
    eirene = [component("e", "poloidal", [[0, 0]], 1, [[1, 0]])]
    genex = [component("g", "poloidal", [[0, 0]], 1, [[1, 0]])]
    native = {"poloidal": np.array([[3.0]]),
              "radial": np.zeros((1, 1))}

    normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D"],
        radial_flux_scaling=2.0,
    )

    assert fort31["fnax"][0, 0] == pytest.approx(3.0)


def test_normalization_rejects_implausible_factor():
    eirene = [component("e", "radial", [[0, 0]], 1, [[1, 0]])]
    genex = [component("g", "radial", [[0, 0]], 1, [[1, 0]])]
    fort31 = {"fnax": np.zeros((1, 1)), "fnay": np.ones((1, 1))}
    native = {"poloidal": np.zeros((1, 1)), "radial": np.full((1, 1), 3.0)}
    with pytest.raises(ValueError, match="outside"):
        normalize_eirene_particle_fluxes(
            fort31, eirene, native, genex, [([0], [0])], ["D"]
        )


def test_negligible_outward_flux_can_scale_to_zero_without_changing_inward():
    fort31 = {"fnax": np.zeros((1, 2)),
              "fnay": np.array([[0.5, -100.0]])}
    eirene = [component("e", "radial", [[0, 0], [0, 1]], 1,
                        [[1, 0], [1, 1]])]
    genex = [component("g", "radial", [[0, 0], [0, 1]], 1,
                       [[1, 0], [1, 1]])]
    native = {"poloidal": np.zeros((1, 2)),
              "radial": np.array([[0.0, -80.0]])}

    records = normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D"],
    )

    np.testing.assert_allclose(fort31["fnay"], [[0.0, -100.0]])
    assert records[0]["factor"] == 0.0
    assert records[0]["eirene_inward"] == pytest.approx(100.0)
    assert records[0]["eirene_outward_after"] == 0.0
    assert "negligible EIRENE outward fraction" in records[0]["warning"]


def test_zero_target_rejects_non_negligible_eirene_outward_flux():
    fort31 = {"fnax": np.zeros((1, 2)),
              "fnay": np.array([[2.0, -100.0]])}
    eirene = [component("e", "radial", [[0, 0], [0, 1]], 1,
                        [[1, 0], [1, 1]])]
    genex = [component("g", "radial", [[0, 0], [0, 1]], 1,
                       [[1, 0], [1, 1]])]
    native = {"poloidal": np.zeros((1, 2)),
              "radial": np.array([[0.0, -80.0]])}

    with pytest.raises(ValueError, match="boundary factor 0 outside"):
        normalize_eirene_particle_fluxes(
            fort31, eirene, native, genex, [([0], [0])], ["D"],
        )


def test_scaled_radial_flux_can_have_non_negligible_zero_target():
    fort31 = {"fnax": np.zeros((1, 2)),
              "fnay": np.array([[2.0, -100.0]])}
    eirene = [component("e", "radial", [[0, 0], [0, 1]], 1,
                        [[1, 0], [1, 1]])]
    genex = [component("g", "radial", [[0, 0], [0, 1]], 1,
                       [[1, 0], [1, 1]])]
    native = {"poloidal": np.zeros((1, 2)),
              "radial": np.array([[0.0, -80.0]])}

    records = normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D"],
        radial_flux_scaling=1e-20,
    )

    np.testing.assert_allclose(fort31["fnay"], [[0.0, -1e-18]])
    assert records[0]["factor"] == 0.0
    assert records[0]["eirene_outward_after"] == 0.0
    assert "scaled radial fluxes" in records[0]["warning"]


def test_boundary_misalignment_extends_normalization_factor_tolerance():
    eirene = [component("e", "radial", [[0, 0]], 1, [[1, 0]])]
    genex = [component("g", "radial", [[0, 0]], 1, [[1, 0]])]
    fort31 = {"fnax": np.zeros((1, 1)), "fnay": np.ones((1, 1))}
    native = {"poloidal": np.zeros((1, 1)),
              "radial": np.full((1, 1), 100.0)}

    records = normalize_eirene_particle_fluxes(
        fort31, eirene, native, genex, [([0], [0])], ["D"],
        allow_boundary_misalignment=True,
    )

    assert records[0]["factor"] == pytest.approx(100.0)
    assert fort31["fnay"][0, 0] == pytest.approx(100.0)


def test_boundary_misalignment_still_rejects_factor_above_extended_tolerance():
    eirene = [component("e", "radial", [[0, 0]], 1, [[1, 0]])]
    genex = [component("g", "radial", [[0, 0]], 1, [[1, 0]])]
    fort31 = {"fnax": np.zeros((1, 1)), "fnay": np.ones((1, 1))}
    native = {"poloidal": np.zeros((1, 1)),
              "radial": np.full((1, 1), 101.0)}

    with pytest.raises(ValueError, match=r"outside \[0.01, 100.0\]"):
        normalize_eirene_particle_fluxes(
            fort31, eirene, native, genex, [([0], [0])], ["D"],
            allow_boundary_misalignment=True,
        )


def test_all_zero_eirene_nonzero_genex_is_fatal():
    eirene = [component("e", "radial", [[0, 0]], 1, [[1, 0]])]
    genex = [component("g", "radial", [[0, 0]], 1, [[1, 0]])]
    fort31 = {"fnax": np.zeros((1, 1)), "fnay": np.zeros((1, 1))}
    native = {"poloidal": np.zeros((1, 1)), "radial": np.ones((1, 1))}
    with pytest.raises(ValueError, match="entirely zero"):
        normalize_eirene_particle_fluxes(
            fort31, eirene, native, genex, [([0], [0])], ["D"]
        )


def test_diagnostics_are_machine_readable(tmp_path):
    path = tmp_path / "normalization.json"
    write_normalization_diagnostics(path, [{"factor": 1.1}])
    assert json.loads(path.read_text()) == [{"factor": 1.1}]
