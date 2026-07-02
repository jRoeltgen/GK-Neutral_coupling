import pytest
import numpy as np
from neutral_coupling.common.strata_assigner import StrataAssigner
from neutral_coupling.common import extra_fort_schema
from collections import defaultdict

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


def test_plasma_plasma_strata_lookup_is_case_insensitive():
    sa = make_assigner(
        requested_strata=(11, "SUM"),
        vol_rec={"particle": {"IONS": {11}}, "energy": {}},
    )

    ingest_series(sa, [
        ("particle", "plasma-plasma", "bulk_ions", "ions", np.array([0.0])),
        ("particle", "plasma-plasma", "bulk_ions", "ions", np.array([2.0])),
    ])

    assert get_strata(
        sa, "particle", "plasma-plasma", "ions"
    ) == [11, "SUM"]
    np.testing.assert_array_equal(
        sa.sources["particle"]["plasma-plasma"]["ions"]["SUM"],
        np.array([2.0]),
    )


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
            },"energy":{}
        }
    )

    ingest_series(sa, [
        ("particle", "atom-plasma", "bulk_ions", "D+", 1.1*np.ones((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.zeros((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.full((3,),2.0)),
        ("particle", "plasma-plasma", "bulk_ions", "D+", np.ones((3,))),
        ("particle", "plasma-plasma", "bulk_ions", "D+", np.full((3,),0.5)),
    ])

    total = sa.sum_over_collisions()

    # VR added to SUM
    assert np.allclose(total["particle"]["D+"]["SUM"], np.full(3, 3.0))

    # VR added ONCE to non-SUM strata (through general sum)
    assert np.allclose(total["particle"]["D+"][21], np.full(3, 2.1))

def test_electrons_get_bulk_ion_vr():
    sa = make_assigner(
        vol_rec={
            "particle":{"D+": {21}},"energy":{}
        }
    )

    ingest_series(sa, [
        ("particle", "atom-plasma", "electrons", "ELECTRONS", np.zeros((3,))),
        ("particle", "atom-plasma", "electrons", "ELECTRONS", 1.1*np.ones((3,))),
        ("particle", "atom-plasma", "electrons", "ELECTRONS", 1.2*np.ones((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", 1.3*np.ones((3,))),
        ("particle", "plasma-plasma", "bulk_ions", "D+", 1.4*np.ones((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.zeros((3,))),
        ("particle", "atom-plasma", "bulk_ions", "D+", np.full((3,),2.0)),
    ])

    total = sa.sum_over_collisions()

    # electrons should receive D+ VR
    assert np.allclose(total["particle"]["ELECTRONS"]["SUM"], np.full((3,),2.6))

def test_SUM_excludes_plasma_plasma():
    sa = make_assigner()

    ingest_series(sa, [
        ("particle", "atom-plasma", "electrons", "ELECTRONS", np.zeros((3,))),
        ("particle", "atom-plasma", "electrons", "ELECTRONS", np.ones((3,))),
        ("particle", "atom-plasma", "electrons", "ELECTRONS", np.ones((3,))),
        ("particle", "plasma-plasma", "electrons", "ELECTRONS", np.ones((3,))),
        ("particle", "plasma-plasma", "electrons", "ELECTRONS", np.zeros((3,))),
        ("particle", "plasma-plasma", "electrons", "ELECTRONS", np.full((3,),2.0)),
    ])

    total = sa.sum_over_collisions()

    # plasma-plasma must not appear in SUM
    assert np.allclose(total["particle"]["ELECTRONS"]["SUM"], np.ones(3,))

def test_Te_Ti_scaling_energy():
    sa = make_assigner(
        vol_rec={
            "particle":{},"energy":{"D+": {21}},
        }
    )

    ingest_series(sa, [
        ("energy", "atom-plasma", "electrons", "ELECTRONS", np.zeros((3,))),
        ("energy", "atom-plasma", "electrons", "ELECTRONS", np.ones((3,))),
        ("energy", "atom-plasma", "electrons", "ELECTRONS", np.ones((3,))),
        ("energy", "atom-plasma", "bulk_ions", "D+", 3.3*np.ones((3,))),
        ("energy", "plasma-plasma", "bulk_ions", "D+", np.ones((3,))),
    ])

    Te = 10.0
    Ti = 2.0  # scaling = 5

    total = sa.sum_over_collisions(Te=Te, Ti=Ti)

    expected = np.ones(3) + Te/Ti*np.ones(3)
    assert np.allclose(total["energy"]["ELECTRONS"]["SUM"], expected)

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
                     "plasma-plasma": {"e-": "electrons", "D+": "bulk_ions"}}
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
                   "plasma-plasma": {"e-": "electrons", "D+": "bulk_ions"}}
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

    # SUM should include VR from stratum 21
    assert np.allclose(total["particle"]["D+"]["SUM"], np.ones(3))

    # non-SUM stratum unchanged
    assert np.allclose(total["particle"]["D+"][21], np.ones(3))

def test_apply_normal_species_vr_plasma_plasma_excluded():
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

    # SUM should ONLY include plasma-plasma VR
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
        suppress_warning=True
    )

    # VR not added to SUM (momentum VR doesn't exist)
    assert np.allclose(total["momentum"]["D+"]["SUM"], np.zeros(3))
    # No warning printed for momentum (we can track the warned set)
    assert ("D+", "SUM") not in warned


# -------------------------
# Integrated test
# -------------------------
def test_full_set_of_parameters():
    req_strata = ["SUM", 11]
    species = {"atoms": ["D", "ATOMS"],
               "molecules": ["D2", "MOLECULES"],
               "test_ions": ["D2+", "TEST IONS"],
               "bulk_ions": ["D+"],
               "electrons": ["ELECTRONS"]}
    vol_rec = {"particle": {"D" :11,
                           "D+":11},
                "energy" : {"ATOMS":11,
                            "D+": 11}}

    sa = StrataAssigner(req_strata, species, vol_rec)
    # Reorder req_strata
    req_strata = [11, "SUM"]
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    values2 = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    MMAP = extra_fort_schema.MOMENT_MAP
    CMAP = extra_fort_schema.COLLISION_MAP
    PCMAP = extra_fort_schema.PARTICLE_CLASS_MAP
    EXCLUDE = {
        (MMAP["1"], CMAP["0"], PCMAP["2"]), #102
        (MMAP["1"], CMAP["0"], PCMAP["3"]), #103
        (MMAP["1"], CMAP["2"], PCMAP["2"]), #122
        (MMAP["1"], CMAP["4"], PCMAP["0"]), #140
        (MMAP["1"], CMAP["4"], PCMAP["3"]), #143
        (MMAP["3"], CMAP["0"], PCMAP["2"]), #302
        (MMAP["3"], CMAP["0"], PCMAP["3"]), #303
        (MMAP["3"], CMAP["2"], PCMAP["2"]), #322
        (MMAP["3"], CMAP["2"], PCMAP["3"]), #323
        (MMAP["3"], CMAP["4"], PCMAP["0"]), #340
        (MMAP["3"], CMAP["4"], PCMAP["3"]), #343
    }
    KEEP = {
        (CMAP["0"], PCMAP["5"]), #205
        (CMAP["1"], PCMAP["5"]), #215
        (CMAP["2"], PCMAP["5"]), #225
    }
    sEXCLUDE = {
        (MMAP["1"], CMAP["4"], PCMAP["2"], 11), # 142
        (MMAP["3"], CMAP["4"], PCMAP["2"], 11), # 342
    }
    #assigner.ingest(moment, collision, pclass, species, "arb", val)
    debug = True
    for mom_key, mom in MMAP.items():
        if mom == "N/A":
            continue
        for coll_key, coll in CMAP.items():
            if coll in (CMAP["3"], CMAP["5"]):
                continue
            for pclass_key, pclass in PCMAP.items():

                if pclass in (PCMAP["4"], PCMAP["6"]):
                    continue
                if (mom, coll, pclass) in EXCLUDE:
                    continue
                if mom == MMAP["2"] and (coll, pclass) not in KEEP:
                    continue
                units = "fort." + str(mom_key) + str(coll_key) + str(pclass_key)
                for sp_key, sp in enumerate(species.get(pclass, [])):
                    if sp in extra_fort_schema.PSEUDO_SPECIES.values() and mom == "particle":
                        continue
                    temp_dict = extra_fort_schema.PSEUDO_SPECIES.copy()
                    temp_dict.update({"bulk_ions":"D+","electrons":"ELECTRONS"})
                    if sp not in temp_dict.values() and mom == "energy":
                        continue
                    for idx, strata in enumerate(req_strata):
                        if (mom, coll, pclass, strata) in sEXCLUDE:
                            continue
                        arr = np.array((mom_key,coll_key,pclass_key,sp_key, idx), dtype=float)
                        sa.ingest(mom, coll, pclass, sp, units, arr, debug=debug)
                        values[mom][coll][sp][strata] = arr
                        values2[units][sp][idx] = arr

    sa.finalize()
    # Check number of keys
    assert len(values2.keys()) == 32

    ##### Check read in properly #####
    for mom, coll_dict in values.items():
        for coll, species_dict in coll_dict.items():
            for species, strata_dict in species_dict.items():
                for stratum, expected in strata_dict.items():
                    found = sa.sources[mom][coll][species][stratum]
                    assert np.allclose(expected, found)

    ##### Check sums #####
    Te = 3
    Ti = 2
    factor = Te/Ti
    total = sa.sum_over_collisions(Te=Te, Ti=Ti)

    # Check momentum "SUM" (205,215,225)
    expected = np.array((6, 3, 15, 0, 3))
    found = total["momentum"]["D+"]["SUM"]
    assert np.allclose(expected, found)

    # Check "SUM" without volume recombination (112)
    expected = np.array((1, 1, 2, 0, 1))
    found = total["particle"]["D2"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-electron "SUM" with (particle) vol. recombination
    # (105, 115, 125, 145)
    expected = np.array((4, 7, 20, 0, 3))
    found = total["particle"]["D+"]["SUM"]
    assert np.allclose(expected, found)

    # Check electron "SUM" with (particle) vol. recombination
    # (100, 110, 120, 145)
    expected = np.array((4, 7, 5, 0, 3))
    found = total["particle"]["ELECTRONS"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-electron/"real particle" "SUM" with (energy) vol. recombination
    # (305, 315, 325, 345)
    expected = np.array((12, 7, 20, 0, 3))
    found = total["energy"]["D+"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-electron/"pseudo particle" "SUM" with (energy) vol. recombination
    # (301, 311, 321, 341)
    expected = np.array((12, 7, 4, 4, 3))
    found = total["energy"]["ATOMS"]["SUM"]
    assert np.allclose(expected, found)

    # Check electron "SUM" with (energy) vol. recombination
    # (300, 310, 320, 2*345)
    expected = np.array((9+factor*3, 3+factor*4, factor*5, 0, 3))
    found = total["energy"]["ELECTRONS"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-"SUM" strata (i.e. strata 11)
        # (105, 115, 125, 145)
    expected = np.array((4, 7, 20, 0, 0))
    found = total["particle"]["D+"][11]
    assert np.allclose(expected, found)

# -------------------------
# Integrated test
# -------------------------
def test_source_restructure():
    req_strata = ["SUM", 11]
    species = {"atoms": ["D", "ATOMS"],
            "molecules": ["D2", "MOLECULES"],
            "test_ions": ["D2+", "TEST IONS"],
            "bulk_ions": ["D+"],
            "electrons": ["ELECTRONS"]}
    vol_rec = {"particle": {"D" :11,
                        "D+":11},
                "energy" : {"ATOMS":11,
                            "D+": 11}}

    sa = StrataAssigner(req_strata, species, vol_rec)
    # Reorder req_strata
    req_strata = [11, "SUM"]
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    MMAP = extra_fort_schema.MOMENT_MAP
    CMAP = extra_fort_schema.COLLISION_MAP
    PCMAP = extra_fort_schema.PARTICLE_CLASS_MAP
    EXCLUDE = {
        (MMAP["1"], CMAP["0"], PCMAP["2"]), #102
        (MMAP["1"], CMAP["0"], PCMAP["3"]), #103
        (MMAP["1"], CMAP["2"], PCMAP["2"]), #122
        (MMAP["1"], CMAP["4"], PCMAP["0"]), #140
        (MMAP["1"], CMAP["4"], PCMAP["3"]), #143
        (MMAP["3"], CMAP["0"], PCMAP["2"]), #302
        (MMAP["3"], CMAP["0"], PCMAP["3"]), #303
        (MMAP["3"], CMAP["2"], PCMAP["2"]), #322
        (MMAP["3"], CMAP["2"], PCMAP["3"]), #323
        (MMAP["3"], CMAP["4"], PCMAP["0"]), #340
        (MMAP["3"], CMAP["4"], PCMAP["3"]), #343
    }
    KEEP = {
        (CMAP["0"], PCMAP["5"]), #205
        (CMAP["1"], PCMAP["5"]), #215
        (CMAP["2"], PCMAP["5"]), #225
    }
    sEXCLUDE = {
        (MMAP["1"], CMAP["4"], PCMAP["2"], 11), # 142
        (MMAP["3"], CMAP["4"], PCMAP["2"], 11), # 342
        (MMAP["1"], CMAP["2"], PCMAP["0"], 11), # 120 "normally should print"
    }
    #assigner.ingest(moment, collision, pclass, species, "arb", val)
    debug = True
    for mom_key, mom in MMAP.items():
        if mom == "N/A":
            continue
        for coll_key, coll in CMAP.items():
            if coll in (CMAP["3"], CMAP["5"]):
                continue
            for pclass_key, pclass in PCMAP.items():
                if pclass in (PCMAP["4"], PCMAP["6"]):
                    continue
                if (mom, coll, pclass) in EXCLUDE:
                    continue
                if mom == MMAP["2"] and (coll, pclass) not in KEEP:
                    continue
                units = "fort." + str(mom_key) + str(coll_key) + str(pclass_key)
                for sp_key, sp in enumerate(species.get(pclass, [])):
                    if sp in extra_fort_schema.PSEUDO_SPECIES.values() and mom == "particle":
                        continue
                    temp_dict = extra_fort_schema.PSEUDO_SPECIES.copy()
                    temp_dict.update({"bulk_ions":"D+","electrons":"ELECTRONS"})
                    if sp not in temp_dict.values() and mom == "energy":
                        continue
                    for idx, strata in enumerate(req_strata):
                        if (mom, coll, pclass, strata) in sEXCLUDE:
                            continue
                        arr = np.array((mom_key,coll_key,pclass_key,sp_key, idx), dtype=float)
                        sa.ingest(mom, coll, pclass, sp, units, arr, debug=debug)
                        values[mom][coll][sp][strata] = arr

    default_coll_to_adjust = {}
    default_coll_to_adjust["testion-plasma"] = True
    sa.finalize(collisions_to_adjust=default_coll_to_adjust, print_info=True)

    ##### Check read in properly #####
    for mom, coll_dict in values.items():
        for coll, species_dict in coll_dict.items():
            for species, strata_dict in species_dict.items():
                for stratum, expected in strata_dict.items():
                    found = sa.sources[mom][coll][species][stratum]
                    assert np.allclose(expected, found)

    ##### Check sums #####
    Te = 3
    Ti = 2
    factor = Te/Ti
    total = sa.sum_over_collisions(Te=Te, Ti=Ti)

    # Check momentum "SUM" (205,215,225)
    expected = np.array((6, 3, 15, 0, 3))
    found = total["momentum"]["D+"]["SUM"]
    assert np.allclose(expected, found)

    # Check "SUM" without volume recombination (112)
    expected = np.array((1, 1, 2, 0, 1))
    found = total["particle"]["D2"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-electron "SUM" with (particle) vol. recombination
    # (105, 115, 125, 145)
    expected = np.array((4, 7, 20, 0, 3))
    found = total["particle"]["D+"]["SUM"]
    assert np.allclose(expected, found)

    # Check electron "SUM" with (particle) vol. recombination
    # (100, 110, 120, 145)
    expected = np.array((4, 7, 5, 0, 3))
    found = total["particle"]["ELECTRONS"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-electron/"real particle" "SUM" with (energy) vol. recombination
    # (305, 315, 325, 345)
    expected = np.array((12, 7, 20, 0, 3))
    found = total["energy"]["D+"]["SUM"]
    assert np.allclose(expected, found)

    # Check non-electron/"pseudo particle" "SUM" with (energy) vol. recombination
    # (301, 311, 321, 341)
    expected = np.array((12, 7, 4, 4, 3))
    found = total["energy"]["ATOMS"]["SUM"]
    assert np.allclose(expected, found)

    # Check electron "SUM" with (energy) vol. recombination
    # (300, 310, 320, 2*345)
    expected = np.array((9+factor*3, 3+factor*4, factor*5, 0, 3))
    found = total["energy"]["ELECTRONS"]["SUM"]
    assert np.allclose(expected, found)
