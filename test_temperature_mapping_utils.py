import numpy as np
import pytest

from temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
    resolve_temperature_label,
    split_ionization_cx,
    linear_temperature_mapper,
    atom_plasma_cx_mapper,
)


def test_default_sparse_temperature_handler_fills_at_threshold():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    result = default_sparse_temperature_handler(
        np.array([5.0, np.nan]),
        np.array([1.0, 0.0]),
        points,
        threshold=0.5,
    )

    np.testing.assert_array_equal(result, np.array([5.0, 5.0]))


@pytest.mark.parametrize(
    ("density", "threshold"),
    [
        (np.array([0.0, 0.0]), 0.5),
        (np.array([1.0, 0.0, 0.0]), 0.5),
    ],
)
def test_default_sparse_temperature_handler_omits_too_sparse_data(
    density, threshold
):
    points = np.column_stack([np.arange(density.size), np.zeros(density.size)])
    temperature = np.where(density > 0, 5.0, np.nan)

    assert default_sparse_temperature_handler(
        temperature, density, points, threshold=threshold
    ) is None


def test_default_sparse_temperature_handler_validates_inputs():
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="between 0 and 1"):
        default_sparse_temperature_handler(
            np.ones(2), np.ones(2), points, threshold=1.1
        )
    with pytest.raises(ValueError, match="same shape"):
        default_sparse_temperature_handler(
            np.ones(1), np.ones(2), points, threshold=0.5
        )
    with pytest.raises(ValueError, match="triangle_points"):
        default_sparse_temperature_handler(
            np.ones(2), np.ones(2), np.ones((2, 3)), threshold=0.5
        )


@pytest.mark.parametrize(
    ("base_label", "species", "temperatures", "expected"),
    [
        ("Ti", "D+", {"Ti_D+": 1, "Ti": 2}, "Ti_D+"),
        ("Tn", "D+", {"Tn_D": 1, "Tn_D+": 2, "Tn": 3}, "Tn_D"),
        ("Tm", "D2", {"Tm": 1}, "Tm"),
        ("Tti", "D2+", {}, None),
    ],
)
def test_resolve_temperature_label(
    base_label, species, temperatures, expected
):
    assert (
        resolve_temperature_label(base_label, species, temperatures)
        == expected
    )


def test_default_pseudo_temperature_handler_is_single_species_only():
    temperature = np.array([1.0, 2.0])
    density = np.array([[3.0], [4.0]])

    pseudo_temperature, pseudo_density = (
        default_single_species_pseudo_temperature_handler(
            [temperature],
            density,
            particle_class="atoms",
            species_labels=["D"],
        )
    )
    np.testing.assert_array_equal(pseudo_temperature, temperature)
    np.testing.assert_array_equal(pseudo_density, density[:, 0])

    with pytest.raises(ValueError, match="only valid for one species"):
        default_single_species_pseudo_temperature_handler(
            [temperature, temperature],
            np.column_stack([density, density]),
            particle_class="atoms",
            species_labels=["D", "T"],
        )

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def simple_case():
    return {
        "Sp": np.array([1.0]),
        "Scx_true": np.array([0.5]),
        "Tn": np.array([3.0]),
        "Ti": np.array([1.0]),
    }


@pytest.fixture
def vector_case():
    return {
        "Sp": np.array([1.0, 2.0, 0.5]),
        "Scx_true": np.array([0.5, 1.0, 0.2]),
        "Tn": np.array([3.0, 4.0, 2.0]),
        "Ti": np.array([1.0, 2.0, 1.5]),
    }


# ============================================================
# 1. split_ionization_cx — simplified with fixtures
# ============================================================

def test_energy_conservation(simple_case):
    Sp = simple_case["Sp"]
    Scx_true = simple_case["Scx_true"]
    Tn = simple_case["Tn"]
    Ti = simple_case["Ti"]

    St = 1.5 * (Sp * Tn + Scx_true * (Tn - Ti))

    _, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti)

    np.testing.assert_allclose(E_ion + E_cx, St, rtol=1e-12, atol=0)


def test_known_solution(simple_case):
    Sp = simple_case["Sp"]
    Scx_true = simple_case["Scx_true"]
    Tn = simple_case["Tn"]
    Ti = simple_case["Ti"]

    St = 1.5 * (Sp * Tn + Scx_true * (Tn - Ti))

    Scx, *_ = split_ionization_cx(St, Sp, Tn, Ti)

    np.testing.assert_allclose(Scx, Scx_true, rtol=1e-12, atol=0)


def test_vectorized_consistency(vector_case):
    Sp = vector_case["Sp"]
    Scx_true = vector_case["Scx_true"]
    Tn = vector_case["Tn"]
    Ti = vector_case["Ti"]

    St = 1.5 * (Sp * Tn + Scx_true * (Tn - Ti))

    Scx, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti)

    np.testing.assert_allclose(Scx, Scx_true, rtol=1e-12, atol=0)
    np.testing.assert_allclose(E_ion + E_cx, St, rtol=1e-12, atol=0)


def test_handles_equal_temperatures():
    St = np.array([9.0])
    Sp = np.array([2.0])
    Tn = np.array([3.0])
    Ti = np.array([3.0])

    Scx, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti)

    assert np.all(np.isfinite(Scx))
    np.testing.assert_allclose(Scx, 0.0, atol=0)
    np.testing.assert_allclose(E_cx, 0.0, atol=0)


def test_partial_mask_behavior():
    St = np.array([10.0, 10.0])
    Sp = np.array([2.0, 2.0])
    Tn = np.array([3.0, 3.0])
    Ti = np.array([1.0, 3.0])  # second entry degenerate

    Scx, _, _ = split_ionization_cx(St, Sp, Tn, Ti)

    assert Scx[1] == 0.0
    assert np.isfinite(Scx[0])


# ============================================================
# 2. linear_temperature_mapper
# ============================================================

def test_atom_plasma_mapper_returns_new_schema():
    from temperature_mapping_utils import atom_plasma_cx_mapper

    out = atom_plasma_cx_mapper(
        mom="energy",
        species="D",
        strata_dict={"SUM": np.array([10.0])},
        temperature_values={"Tn": np.array([3.0]), "Ti": np.array([1.0])},
        context={"particle_sources": {"D": np.array([1.0])}},
    )

    assert "values" in out
    assert "conserved_and_constrained" in out
    assert set(out["values"].keys()) == {"Tn", "Ti"}

def test_linear_temperature_mapper():
    mapper = linear_temperature_mapper("Ti")

    arr = np.array([1.0, 2.0])
    strata_dict = {"SUM": arr}

    result = mapper(
        mom="density",
        species="D",
        strata_dict=strata_dict,
        temperature_values={},
        context={},
    )

    assert "values" in result
    assert "Ti" in result["values"]
    np.testing.assert_array_equal(result["values"]["Ti"], arr)


# ============================================================
# 3. atom_plasma_cx_mapper — behavior
# ============================================================

def test_atom_plasma_mapper_non_energy():
    arr = np.array([5.0, 6.0])

    result = atom_plasma_cx_mapper(
        mom="density",
        species="D",
        strata_dict={"SUM": arr},
        temperature_values={"Tn": 3.0, "Ti": 1.0},
        context={"particle_sources": {"D": np.array([1.0, 1.0])}},
    )

    np.testing.assert_array_equal(result["values"]["Tn"], arr)


def test_atom_plasma_mapper_energy_split(simple_case):
    Sp = simple_case["Sp"]
    Scx_true = simple_case["Scx_true"]
    Tn = simple_case["Tn"]
    Ti = simple_case["Ti"]

    St = 1.5 * (Sp * Tn + Scx_true * (Tn - Ti))

    result = atom_plasma_cx_mapper(
        mom="energy",
        species="D",
        strata_dict={"SUM": St},
        temperature_values={"Tn": Tn, "Ti": Ti},
        context={"particle_sources": {"D": Sp}},
    )

    E_ion = result["values"]["Tn"]
    E_cx = result["values"]["Ti"]

    np.testing.assert_allclose(E_ion + E_cx, St, rtol=1e-12, atol=0)


# ============================================================
# 4. Physics-consistency regression tests
# ============================================================

def test_forward_inverse_consistency(vector_case):
    kB = 1.0

    Sp = vector_case["Sp"]
    Scx_true = vector_case["Scx_true"]
    Tn = vector_case["Tn"]
    Ti = vector_case["Ti"]

    St = 1.5 * kB * (Sp * Tn + Scx_true * (Tn - Ti))

    Scx, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti, kB=kB)

    np.testing.assert_allclose(Scx, Scx_true, rtol=1e-12, atol=0)
    np.testing.assert_allclose(E_ion, 1.5 * kB * Sp * Tn, rtol=1e-12, atol=0)
    np.testing.assert_allclose(E_cx, 1.5 * kB * Scx_true * (Tn - Ti), rtol=1e-12, atol=0)
    np.testing.assert_allclose(E_ion + E_cx, St, rtol=1e-12, atol=0)


def test_no_cx_limit():
    Sp = np.array([1.0])
    Tn = np.array([3.0])
    Ti = np.array([1.0])

    St = 1.5 * (Sp * Tn)

    Scx, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti)

    np.testing.assert_allclose(Scx, 0.0, atol=0)
    np.testing.assert_allclose(E_cx, 0.0, atol=0)
    np.testing.assert_allclose(E_ion, St, rtol=1e-12, atol=0)


def test_pure_cx_limit():
    Sp = np.array([0.0])
    Scx_true = np.array([1.5])
    Tn = np.array([4.0])
    Ti = np.array([2.0])

    St = 1.5 * (Scx_true * (Tn - Ti))

    Scx, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti)

    np.testing.assert_allclose(Scx, Scx_true, rtol=1e-12, atol=0)
    np.testing.assert_allclose(E_ion, 0.0, atol=0)
    np.testing.assert_allclose(E_cx, St, rtol=1e-12, atol=0)


def test_thermal_equilibrium_limit():
    Sp = np.array([1.0])
    Tn = Ti = np.array([3.0])

    St = 1.5 * Sp * Tn

    Scx, E_ion, E_cx = split_ionization_cx(St, Sp, Tn, Ti)

    np.testing.assert_allclose(E_cx, 0.0, atol=0)
    np.testing.assert_allclose(E_ion, St, rtol=1e-12, atol=0)


# ============================================================
# 5. Mapper-level physics regression
# ============================================================

def test_atom_plasma_mapper_physics_consistency(simple_case):
    Sp = simple_case["Sp"]
    Scx_true = simple_case["Scx_true"]
    Tn = simple_case["Tn"]
    Ti = simple_case["Ti"]

    St = 1.5 * (Sp * Tn + Scx_true * (Tn - Ti))

    result = atom_plasma_cx_mapper(
        mom="energy",
        species="D",
        strata_dict={"SUM": St},
        temperature_values={"Tn": Tn, "Ti": Ti},
        context={"particle_sources": {"D": Sp}},
    )

    np.testing.assert_allclose(result["values"]["Tn"], 1.5 * Sp * Tn, rtol=1e-12, atol=0)
    np.testing.assert_allclose(result["values"]["Ti"], 1.5 * Scx_true * (Tn - Ti), rtol=1e-12, atol=0)
