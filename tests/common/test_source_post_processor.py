import numpy as np
import pytest
from neutral_coupling.common.source_post_processor import SourcePostProcessor

@pytest.fixture
def simple_sources():
    """
    Minimal consistent dataset with:
    - particle source (needed for CX)
    - energy source (to be split)
    """
    Sp = np.array([1.0])
    Scx = np.array([0.5])
    Tn = np.array([3.0])
    Ti = np.array([1.0])

    # Construct consistent energy source
    St = 1.5 * (Sp * Tn + Scx * (Tn - Ti))

    return {
        "particle": {
            "atom-plasma": {
                "D": {"SUM": Sp}
            }
        },
        "energy": {
            "atom-plasma": {
                "D": {"SUM": St}
            }
        }
    }

@pytest.fixture
def temperature_values():
    return {
        "Tn": np.array([3.0]),
        "Ti": np.array([1.0]),
    }

@pytest.fixture
def broken_linear_mapper():
    def mapper(*args, **kwargs):
        return {"values":
                    {"Ti": np.array([999.0])},
                "conserved_and_constrained": True,
                }
    return mapper

def test_transpose_sources(simple_sources, temperature_values):
    proc = SourcePostProcessor(simple_sources, temperature_values)

    out = proc.transpose_sources()

    assert "particle" in out
    assert "D" in out["particle"]
    assert "atom-plasma" in out["particle"]["D"]

    np.testing.assert_array_equal(
        out["particle"]["D"]["atom-plasma"]["SUM"],
        simple_sources["particle"]["atom-plasma"]["D"]["SUM"]
    )

def test_context_extraction(simple_sources, temperature_values):
    proc = SourcePostProcessor(simple_sources, temperature_values)
    proc.transpose_sources()

    context = proc._get_context()

    assert "D" in context["particle_sources"]

    np.testing.assert_array_equal(
        context["particle_sources"]["D"],
        simple_sources["particle"]["atom-plasma"]["D"]["SUM"]
    )

def test_regroup_atom_plasma_energy(simple_sources, temperature_values):
    proc = SourcePostProcessor(simple_sources, temperature_values)

    sources_T, temp = proc.regroup_by_temperature()

    Sp = simple_sources["particle"]["atom-plasma"]["D"]["SUM"]
    Tn = temperature_values["Tn"]
    Ti = temperature_values["Ti"]

    # Expected physics
    E_ion = 1.5 * Sp * Tn
    E_cx  = 1.5 * 0.5 * (Tn - Ti)

    np.testing.assert_allclose(
        sources_T["energy"]["D"]["Tn"], E_ion, rtol=1e-12, atol=0
    )
    np.testing.assert_allclose(
        sources_T["energy"]["D"]["Ti"], E_cx, rtol=1e-12, atol=0
    )

def test_atom_plasma_decomposition(simple_sources, temperature_values):
    proc = SourcePostProcessor(simple_sources, temperature_values)

    sources_T, _ = proc.regroup_by_temperature()

    Sp = simple_sources["particle"]["atom-plasma"]["D"]["SUM"]
    Tn = temperature_values["Tn"]
    Ti = temperature_values["Ti"]

    # independent reconstruction check
    reconstructed = (
        sources_T["energy"]["D"]["Tn"]
        + sources_T["energy"]["D"]["Ti"]
    )

    St = simple_sources["energy"]["atom-plasma"]["D"]["SUM"]

    np.testing.assert_allclose(reconstructed, St, rtol=1e-12, atol=0)

@pytest.fixture
def multi_collision_sources():
    arr1 = np.array([2.0])
    arr2 = np.array([3.0])

    return {
        "density": {
            "plasma-plasma": {
                "D": {"SUM": arr1}
            },
            "atom-plasma": {
                "D": {"SUM": arr2}
            }
        }
    }


def test_aggregation_across_collisions(multi_collision_sources):
    temps = {"Ti": np.array([1.0]), "Tn": np.array([1.0])}

    proc = SourcePostProcessor(multi_collision_sources, temps)

    sources_T, _ = proc.regroup_by_temperature()

    # plasma-plasma → Ti
    # atom-plasma (non-energy) → Tn
    np.testing.assert_array_equal(
        sources_T["density"]["D"]["Ti"],
        np.array([2.0])
    )
    np.testing.assert_array_equal(
        sources_T["density"]["D"]["Tn"],
        np.array([3.0])
    )

def test_shape_mismatch_raises(simple_sources, temperature_values):
    bad = simple_sources.copy()

    bad["energy"]["atom-plasma"]["D"]["SUM"] = np.array([1.0, 2.0])

    proc = SourcePostProcessor(bad, temperature_values)

    with pytest.raises(ValueError):
        proc.regroup_by_temperature()

def test_linear_conservation_violation(broken_linear_mapper):
    sources = {
        "density": {
            "plasma-plasma": {
                "D": {"SUM": np.array([10.0])}
            }
        }
    }

    temperature_values = {"Ti": np.array([1.0])}

    proc = SourcePostProcessor(
        sources,
        temperature_values,
        collision_mappers={"plasma-plasma": broken_linear_mapper}
    )

    with pytest.raises(AssertionError):
        proc.regroup_by_temperature()


def test_composite_temperature_labels_are_used_for_d_plus():
    Sp = np.array([1.0])
    Tn = np.array([3.0])
    Ti = np.array([1.0])
    St = 1.5 * Sp * Tn
    sources = {
        "particle": {"atom-plasma": {"D+": {"SUM": Sp}}},
        "energy": {"atom-plasma": {"D+": {"SUM": St}}},
    }

    proc = SourcePostProcessor(
        sources,
        {"Tn_D": Tn, "Ti_D+": Ti},
    )
    sources_T, used_temperatures = proc.regroup_by_temperature()

    assert set(sources_T["energy"]["D+"]) == {"Tn_D", "Ti_D+"}
    assert set(used_temperatures) == {"Tn_D", "Ti_D+"}


def test_missing_temperature_uses_implicit_channel():
    sources = {
        "particle": {
            "molecule-plasma": {
                "D2+": {"SUM": np.array([2.0])}
            }
        }
    }

    proc = SourcePostProcessor(sources, {})
    sources_T, used_temperatures = proc.regroup_by_temperature()

    np.testing.assert_array_equal(
        sources_T["particle"]["D2+"][None], np.array([2.0])
    )
    assert used_temperatures == {}


def test_default_mapper_rejects_multi_species_but_explicit_mapper_is_trusted():
    temperatures = {
        "Tn_D": np.array([1.0]),
        "Tn_T": np.array([1.0]),
        "Ti_D+": np.array([1.0]),
        "Ti_T+": np.array([1.0]),
    }

    with pytest.raises(ValueError, match="built-in CX mapper"):
        SourcePostProcessor({}, temperatures)

    custom = {"atom-plasma": lambda **kwargs: {"values": {None: np.array([1.0])}}}
    SourcePostProcessor({}, temperatures, collision_mappers=custom)
