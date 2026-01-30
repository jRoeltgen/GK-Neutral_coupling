import pytest
import numpy as np
import pdb
from StrataAssigner import StrataAssigner


def make_assigner(
    requested_strata=(21, 22, "SUM"),
    vol_rec=None,
    debug=False
):
    if vol_rec is None:
        vol_rec = {"particle":{}, "energy":{}}

    return StrataAssigner(
        requested_strata=list(requested_strata),
        vol_rec_mapping=vol_rec,
        debug=debug
    )


# -------------------------
# helpers
# -------------------------

def ingest_series(assigner, entries):
    """
    entries = list of tuples:
      (moment, collision, particle_class, species, value)
    """
    for moment, collision, pclass, species, val in entries:
        assigner.ingest(moment, collision, pclass, species, "arb", val)


def get_strata(assigner, moment, collision, species):
    return list(assigner.sources[moment][collision][species].keys())


# ============================================================
# BASIC STRATA CYCLING
# ============================================================

def test_basic_strata_sequence():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "atoms", "D", np.array([1.0])),
        ("particle", "atom-plasma", "atoms", "D", np.array([2.0])),
        ("particle", "atom-plasma", "atoms", "D", np.array([3.0])),
    ])

    strata = get_strata(sa, "particle", "atom-plasma", "D")
    assert strata == [21, 22, "SUM"]


# ============================================================
# PLASMA–PLASMA VOLUME RECOMBINATION LOGIC
# ============================================================

def test_plasma_plasma_volume_recombination_assignment():
    sa = make_assigner(
        vol_rec={
            "particle":{
                "D+": {21},
                "T+": {22},
            },"energy":{}
        }
    )

    ingest_series(sa, [
        ("particle", "plasma-plasma", "bulk_ions", "D+", np.array([1.0])),
        ("particle", "plasma-plasma", "bulk_ions", "T+", np.array([1.0])),
        ("particle", "plasma-plasma", "bulk_ions", "D+", np.array([2.0])),
        ("particle", "plasma-plasma", "bulk_ions", "T+", np.array([2.0])),
    ])

    d_strata = get_strata(sa, "particle", "plasma-plasma", "D+")
    t_strata = get_strata(sa, "particle", "plasma-plasma", "T+")

    assert d_strata == [21, "SUM"]
    assert t_strata == [22, "SUM"]


# ============================================================
# STRATUM SKIPPING (BUG REGRESSION TEST)
# ============================================================

def test_plasma_plasma_stratum_skipping():
    sa = make_assigner(
        requested_strata=[21, 22, "SUM"],
        vol_rec={"particle":{"T+": {22}},"energy":{}}
    )

    ingest_series(sa, [
        ("particle", "plasma-plasma", "bulk_ions", "T+", np.array([1.0])),
        ("particle", "plasma-plasma", "bulk_ions", "T+", np.array([2.0])),
    ])

    strata = get_strata(sa, "particle", "plasma-plasma", "T+")
    assert strata == [22, "SUM"]   # MUST skip 21


# ============================================================
# MULTI-SPECIES INDEPENDENCE
# ============================================================

def test_group_local_indexing():
    sa = make_assigner(
        vol_rec={
            "particle":{
                "D+": {21},
                "T+": {22},
            },"energy":{}
        }
    )

    ingest_series(sa, [
        ("particle", "plasma-plasma", "bulk_ions", "D+", np.array([1.0])),
        ("particle", "plasma-plasma", "bulk_ions", "T+", np.array([1.0])),
        ("particle", "plasma-plasma", "bulk_ions", "D+", np.array([2.0])),
        ("particle", "plasma-plasma", "bulk_ions", "T+", np.array([2.0])),
    ])

    d_strata = get_strata(sa, "particle", "plasma-plasma", "D+")
    t_strata = get_strata(sa, "particle", "plasma-plasma", "T+")

    assert d_strata == [21, "SUM"]
    assert t_strata == [22, "SUM"]


# ============================================================
# SUM_OVER_COLLISIONS
# ============================================================

def test_sum_over_collisions_simple():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "atoms", "D", np.array([1.0])),
        ("particle", "molecule-plasma", "atoms", "D", np.array([2.0])),
        ("particle", "testion-plasma", "atoms", "D", np.array([3.0])),
    ])

    summed_over_collisions = sa.sum_over_collisions()

    summed = summed_over_collisions["particle"]["D"][21]
    assert summed == pytest.approx(6.0)


def test_sum_over_collisions_strata_preserved():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "atoms", "D", np.array([1.0])),   # 21
        ("particle", "atom-plasma", "atoms", "D", np.array([2.0])),   # 22
        ("particle", "molecule-plasma", "atoms", "D", np.array([3.0])),  # 21
        ("particle", "molecule-plasma", "atoms", "D", np.array([4.0])),  # 22
    ])

    summed_over_collisions = sa.sum_over_collisions()

    summed = summed_over_collisions["particle"]["D"][21]
    assert summed == pytest.approx(np.array([1.0 + 3.0]))

    summed = summed_over_collisions["particle"]["D"][22]
    assert summed == pytest.approx(np.array([2.0 + 4.0]))


def test_sum_over_collisions_with_sum_stratum():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "atoms", "D", np.array([1.0])),
        ("particle", "atom-plasma", "atoms", "D", np.array([2.0])),
        ("particle", "atom-plasma", "atoms", "D", np.array([3.0])),
    ])

    summed_over_collisions = sa.sum_over_collisions()

    summed = summed_over_collisions["particle"]["D"]["SUM"]

    assert summed == pytest.approx(np.array([3.0]))

def test_VR_only_adds_to_SUM():
    sa = make_assigner(
        vol_rec={
            "particle":{
                "D+": {21},
                "T+": {22},
            },"energy":{}
        }
    )

    ingest_series(sa, [
        ("particle", "atom-plasma", "bulk_ions", "D+", np.ones((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.zeros((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.full((3,),2.0)),
    ])

    total = sa.sum_over_collisions()

    # VR added to SUM
    assert np.allclose(total["particle"]["D+"]["SUM"], np.full(3, 3.0))

    # VR NOT added to non-SUM strata
    assert np.allclose(total["particle"]["D+"][21], np.full(3, 1.0))

def test_electrons_get_bulk_ion_vr():
    sa = make_assigner(
        vol_rec={
            "particle":{"D+": {21}},"energy":{}
        }
    )

    ingest_series(sa, [
        ("particle", "atom-plasma", "ELECTRONS", "e-", np.zeros((3,))),
        ("particle", "atom-plasma", "ELECTRONS", "e-", 1.1*np.ones((3,))),
        ("particle", "atom-plasma", "ELECTRONS", "e-", 1.2*np.ones((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", 1.3*np.ones((3,))),
        ("particle", "plasma-plasma", "bulk_ions", "D+", 1.4*np.ones((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.zeros((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.full((3,),2.0)),
    ])

    total = sa.sum_over_collisions()

    # electrons should receive D+ VR
    assert np.allclose(total["particle"]["e-"]["SUM"], np.full((3,),2.6))

def test_SUM_excludes_plasma_plasma():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "ELECTRONS", "e-", np.zeros((3,))),
        ("particle", "atom-plasma", "ELECTRONS", "e-", np.ones((3,))),
        ("particle", "atom-plasma", "ELECTRONS", "e-", np.ones((3,))),
        ("particle", "plasma-plasma", "ELECTRONS", "e-", np.ones((3,))),
        ("particle", "plasma-plasma", "ELECTRONS", "e-", np.zeros((3,))),
        ("particle", "plasma-plasma", "ELECTRONS", "e-", np.full((3,),2.0)),
    ])

    total = sa.sum_over_collisions()

    # plasma-plasma must not appear in SUM
    assert np.allclose(total["particle"]["e-"]["SUM"], np.ones(3,))

def test_Te_Ti_scaling_energy():
    sa = make_assigner(
        vol_rec={
            "particle":{},"energy":{"D+": {21}},
        }
    )

    ingest_series(sa, [
        ("energy", "atom-plasma", "ELECTRONS", "e-", np.zeros((3,))),
        ("energy", "atom-plasma", "ELECTRONS", "e-", np.ones((3,))),
        ("energy", "atom-plasma", "ELECTRONS", "e-", np.ones((3,))),
        ("energy", "atom-plasma", "bulk_ions", "D+", 3.3*np.ones((3,))),
        ("energy", "plasma-plasma", "bulk_ions", "D+", np.ones((3,))),
    ])

    Te = 10.0
    Ti = 2.0  # scaling = 5

    total = sa.sum_over_collisions(Te=Te, Ti=Ti)

    expected = np.ones(3) + Te/Ti*np.ones(3)
    assert np.allclose(total["energy"]["e-"]["SUM"], expected)

# ============================================================
# CROSS-MOMENT ISOLATION
# ============================================================

def test_sum_over_collisions_does_not_mix_moments():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "atoms", "D", np.array([1.0])),
        ("energy",   "atom-plasma", "atoms", "D", np.array([10.0])),
    ])

    summed_over_collisions = sa.sum_over_collisions()

    p_sum = summed_over_collisions["particle"]["D"][21]
    e_sum = summed_over_collisions["energy"]["D"][21]

    assert p_sum == pytest.approx(1.0)
    assert e_sum == pytest.approx(10.0)

# -------------------------
# Electron VR helpers
# -------------------------
def test_apply_electron_bulk_vr_particle():
    sa = make_assigner(vol_rec={"particle":{"D+": {21}},"energy":{}})

    # minimal sources
    sa.sources = {
        "particle": {
            "atom-plasma": {
                "D+": {21: np.ones(3)}
            },
            "plasma-plasma":{
                "e-": {"SUM": np.zeros(3)},
                "D+": {21: np.ones(3)}
            }
        }
    }
    sa.particle_class = {
        "particle": {"atom-plasma": {"D+": "bulk_ions"},
                     "plasma-plasma": {"e-": "ELECTRONS", "D+": "bulk_ions"}}
    }

    warned = set()
    total = {"particle": {"e-": {"SUM": np.zeros(3)}}}

    sa._apply_electron_bulk_vr(
        mom="particle",
        coll_dict=["plasma-plasma"],
        total=total,
        vol_attr=sa.volume_recombination,
        Te=1.0,
        Ti=1.0,
        el_species="e-",
        warned_species_strata=warned,
        key_warn=("e-", "SUM"),
        suppress_warning=False
    )

    # This should not affect the sum, so suppress warnings
    sa._apply_electron_bulk_vr(
        mom="particle",
        coll_dict=["atom-plasma"],
        total=total,
        vol_attr=sa.volume_recombination,
        Te=1.0,
        Ti=1.0,
        el_species="e-",
        warned_species_strata=warned,
        key_warn=("e-", "SUM"),
        suppress_warning=True
    )

    # Check that electron SUM got D+ contribution
    assert np.allclose(total["particle"]["e-"]["SUM"], np.ones(3))

def test_apply_electron_bulk_vr_energy_scaling():
    sa = make_assigner(vol_rec={"particle":{},"energy":{"D+": {21}}})

    sa.sources = {
        "energy": {
            "atom-plasma": {
                "D+": {21: 1.1*np.ones(3)}
            },
            "plasma-plasma":{
                "e-": {"SUM": np.zeros(3)},
                "D+": {21: np.ones(3)}
            }
        }
    }
    sa.particle_class = {
        "energy": {"atom-plasma": {"D+": "bulk_ions"},
                   "plasma-plasma": {"e-": "ELECTRONS", "D+": "bulk_ions"}}
    }

    warned = set()
    total = {"energy": {"e-": {"SUM": np.zeros(3)}}}

    Te, Ti = 10.0, 2.0  # factor = 5

    sa._apply_electron_bulk_vr(
        mom="energy",
        coll_dict=["plasma-plasma"],
        total=total,
        vol_attr=sa.volume_recombination["energy"],
        Te=Te,
        Ti=Ti,
        el_species="e-",
        warned_species_strata=warned,
        key_warn=("e-", "SUM"),
        suppress_warning=False
    )

    # Electron SUM should include D+ VR scaled by Te/Ti
    assert np.allclose(total["energy"]["e-"]["SUM"], 5*np.ones(3))

# -------------------------
# Normal species VR helpers
# -------------------------
def test_apply_normal_species_vr_only_sum():
    sa = make_assigner(vol_rec={"particle":{"D+": {21}},"energy":{}})

    sa.sources = {
        "particle": {
            "atom-plasma": {
                "D+": {21: np.ones(3)}
            }
        }
    }
    sa.particle_class = {
        "particle": {"atom-plasma": {"D+": "bulk_ions"}}
    }

    total = {"particle": {"D+": {"SUM": np.zeros(3), 21: np.ones(3)}}}
    warned = set()

    sa._apply_normal_species_vr(
        mom="particle",
        coll_dict=["atom-plasma"],
        species="D+",
        total=total,
        vol_attr=sa.volume_recombination["particle"],
        warned_species_strata=warned,
        key_warn=("D+", "SUM"),
        suppress_warning=False
    )

    # SUM should include VR from stratum 21
    assert np.allclose(total["particle"]["D+"]["SUM"], np.ones(3))

    # non-SUM stratum unchanged
    assert np.allclose(total["particle"]["D+"][21], np.ones(3))

def test_apply_normal_species_vr_plasma_plasma_excluded():
    sa = make_assigner(vol_rec={"particle":{"D+": {21}},"energy":{}})

    sa.sources = {
        "particle": {
            "plasma-plasma": {
                "D+": {21: np.ones(3)}
            }
        }
    }
    sa.particle_class = {
        "particle": {"plasma-plasma": {"D+": "bulk_ions"}}
    }

    total = {"particle": {"D+": {"SUM": np.zeros(3), 21: np.ones(3)}}}
    warned = set()

    sa._apply_normal_species_vr(
        mom="particle",
        coll_dict=["plasma-plasma"],
        species="D+",
        total=total,
        vol_attr=sa.volume_recombination["particle"],
        warned_species_strata=warned,
        key_warn=("D+", "SUM"),
        suppress_warning=False
    )

    # SUM should NOT include plasma-plasma VR
    assert np.allclose(total["particle"]["D+"]["SUM"], np.zeros(3))

# -------------------------
# Warnings suppressed
# -------------------------
def test_apply_vr_warnings_suppressed():
    sa = make_assigner(
        vol_rec={
                "particle":{
                    "D+": {21},
                },
                "energy":{}
            }
        )

    sa.sources = {
        "momentum": {
            "atom-plasma": {
                "D+": {21: np.ones(3)}
            }
        }
    }
    sa.particle_class = {
        "momentum": {"atom-plasma": {"D+": "bulk_ions"}}
    }

    total = {"momentum": {"D+": {"SUM": np.zeros(3), 21: np.ones(3)}}}
    warned = set()

    sa._apply_normal_species_vr(
        mom="momentum",
        coll_dict=["atom-plasma"],
        species="D+",
        total=total,
        vol_attr=sa.volume_recombination["particle"],
        warned_species_strata=warned,
        key_warn=("D+", "SUM"),
        suppress_warning=False
    )

    # VR added to SUM
    assert np.allclose(total["momentum"]["D+"]["SUM"], np.ones(3))
    # No warning printed for momentum (we can track the warned set)
    assert ("D+", "SUM") not in warned
