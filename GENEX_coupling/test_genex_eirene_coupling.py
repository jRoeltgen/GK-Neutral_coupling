import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from genex_eirene_coupling import (main, status, interpolate_all_moments,
                                   check_species_consistency,
                                   interpolate_all_sources_wrapper,
                                   normalize_genex_params)
import numpy as np
import xarray as xr
from collections import defaultdict

@pytest.fixture
def coupling_env(monkeypatch, tmp_path):
    import genex_eirene_coupling as mod

    # ----------------------------
    # Args (baseline)
    # ----------------------------
    args = SimpleNamespace(
        pid=1234,
        eirene_time=10,
        MAX_TIMEOUTS=1,
        SumTemp=True,
        filepattern=str(tmp_path / "out_"),
        genex_time_index_override=False,
        eirene_path=tmp_path,
        genex_path=tmp_path,
    )

    # ----------------------------
    # Deps (DI)
    # ----------------------------
    deps = SimpleNamespace(
        run_eirene=MagicMock(return_value=mod.status.SUCCESS),
        write_nc=MagicMock(),
        killpg=MagicMock(),
        sleep=MagicMock(),
        replace=MagicMock(),
        pid_exists=MagicMock(side_effect=[True, False]),
    )

    # ----------------------------
    # Grid (must support .values)
    # ----------------------------
    grid = MagicMock()
    grid.r_u = xr.DataArray([1.0])
    grid.z_u = xr.DataArray([2.0])

    r_all = grid.r_u
    z_all = grid.z_u
    compute = True

    norm = {"R0": 1.0}

    params = {"params_species":{}}

    # ----------------------------
    # GENEX mocks
    # ----------------------------
    monkeypatch.setattr(
        mod.genex_interface,
        "wait_for_genex_init",
        MagicMock(return_value=(grid, None, params, norm, r_all, z_all, compute)),
    )

    monkeypatch.setattr(
        mod.genex_interface,
        "get_genex_species",
        MagicMock(return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]),
    )

    def make_es_pot(scale):
        return xr.DataArray(
            data=np.ones((4, 10)),
            dims=("phi", "RZ"),
            attrs={"norm":1.0}
        )

    fields = {"es_pot": {"N/A": make_es_pot(1.0)}}

    monkeypatch.setattr(
        mod.genex_interface,
        "load_latest_genex_fields",
        MagicMock(return_value=(fields, 0.001)),
    )

    monkeypatch.setattr(
        mod.genex_interface,
        "toroidal_avg",
        MagicMock(return_value={}),
    )

    # ----------------------------
    # EIRENE mocks
    # ----------------------------
    edat = MagicMock()
    edat.species_names = {"bulk_ions": ["D"]}
    edat.sources = {"D": np.array([1.0])}
    edat.write_ft31 = MagicMock()
    edat.load_extra_forts = MagicMock()

    b2dat = MagicMock()
    b2dat.gmtry = "gmtry"

    monkeypatch.setattr(
        mod,
        "eirene_interface",
        MagicMock(return_value=(edat, b2dat)),
    )

    # ----------------------------
    # Other required patches
    # ----------------------------
    monkeypatch.setattr(mod, "build_triangulation", MagicMock(return_value="tri"))
    monkeypatch.setattr(mod, "interpolate_all_moments", MagicMock(return_value={}))
    monkeypatch.setattr(mod, "interpolate_all_sources", MagicMock(return_value={}))
    monkeypatch.setattr(mod, "prepare_fort31", MagicMock())

    return {
        "args": args,
        "deps": deps,
        "mod": mod,
        "grid": grid,
        "edat": edat,
        "b2dat": b2dat,
        "fields": fields,
        "make_es_pot": make_es_pot,
    }

def test_main_single_iteration_success(coupling_env):
    env = coupling_env

    env["mod"].main(env["args"], deps=env["deps"])

    env["deps"].run_eirene.assert_called_once()
    env["deps"].write_nc.assert_called_once()
    env["deps"].replace.assert_called_once()
    env["deps"].killpg.assert_not_called()

def test_main_timeout_kills_and_raises(coupling_env):
    env = coupling_env

    env["deps"].run_eirene.return_value = env["mod"].status.TIMEOUT

    with pytest.raises(RuntimeError):
        env["mod"].main(env["args"], deps=env["deps"])

    env["deps"].killpg.assert_called_once()

def test_main_retry_and_doubling(coupling_env):
    env = coupling_env

    env["args"].MAX_TIMEOUTS = 3

    env["deps"].run_eirene.side_effect = [
        env["mod"].status.TIMEOUT,
        env["mod"].status.TIMEOUT,
        env["mod"].status.SUCCESS,
    ]

    env["mod"].main(env["args"], deps=env["deps"])

    calls = [c.args[0] for c in env["deps"].run_eirene.call_args_list]
    assert calls == [10, 20, 40]

def test_main_multiple_iterations(coupling_env):
    env = coupling_env

    env["deps"].pid_exists = MagicMock(side_effect=[True, True, True, False])

    env["mod"].genex_interface.load_latest_genex_fields.side_effect = [
        (env["fields"], 0.001),
        (env["fields"], 0.01),
        (env["fields"], 0.1),
    ]

    env["mod"].main(env["args"], deps=env["deps"])

    assert env["deps"].run_eirene.call_count == 3
    assert env["deps"].write_nc.call_count == 3

def test_unnormalize_all_mutates():
    import genex_eirene_coupling as mod

    data = {"field": {"D": 1.0}}

    mod.genex_interface.unnormalize = MagicMock(return_value=2.0)

    mod.unnormalize_all(data)

    assert data["field"]["D"] == 2.0

def test_get_genex_electron_name_none():
    from genex_eirene_coupling import get_genex_electron_name

    species = [SimpleNamespace(name="D", is_electron=False)]

    assert get_genex_electron_name(species) is None

def test_main_sumtemp_false_raises(coupling_env):
    env = coupling_env

    # Override only what matters
    env["args"].SumTemp = False

    with pytest.raises(NotImplementedError, match="SumTemp=False not implemented"):
        env["mod"].main(env["args"], deps=env["deps"])

def test_check_species_consistency_raises():
    eirene_species = ["D"]
    genex_species = [
        SimpleNamespace(name="T", is_electron=False),
        SimpleNamespace(name="e", is_electron=True),
    ]

    with pytest.raises(ValueError) as exc:
        check_species_consistency(eirene_species, genex_species)
    assert "Genex Species T not known to Eirene" in str(exc.value)

def test_interpolate_all_moments_indices(monkeypatch):

    calls = []

    def fake_interp(gmtry, tri, arr, ind):
        calls.append(ind)
        return {"ok": True}

    monkeypatch.setattr("genex_eirene_coupling.interp_moments", fake_interp)

    class FakeValue:
        def __init__(self, arr):
            self.data = arr

    gmtry = "gmtry"
    tri = "tri"

    genex_out = {
        "poloidal_fluxes": {"D": FakeValue(np.ones((3, 3)))},
        "radial_fluxes": {"D": FakeValue(np.ones((3, 3)))},
        "other": {"D": FakeValue(np.ones((3, 3)))},
    }

    result = interpolate_all_moments(gmtry, tri, genex_out)

    assert calls[0] == [0, 2]
    assert calls[1] == [2, 3]
    assert calls[2] == [0, 1, 2, 3]

    assert result["poloidal_fluxes"]["D"] == {"ok": True}


def test_main_tau_must_increase(coupling_env):
    env = coupling_env
    mod = env["mod"]

    env["deps"].pid_exists = MagicMock(side_effect=[True, True, False])

    env["mod"].genex_interface.load_latest_genex_fields.side_effect = [
        (env["fields"], 0.02),
        (env["fields"], 0.01),  # regression
    ]

    mod.main(env["args"], deps=env["deps"])

    # second iteration skipped
    assert env["deps"].run_eirene.call_count == 1

# ----------------------------------------------------------------------
# normalize_genex_params tests
# ----------------------------------------------------------------------

def test_normalize_genex_params_strips_species_names():
    params = {
        "params_species": {
            "names": [" ions ", " ELECTRONS", "impurity  "]
        }
    }

    result = normalize_genex_params(params)

    assert result["params_species"]["names"] == [
        "ions",
        "ELECTRONS",
        "impurity",
    ]


def test_normalize_genex_params_missing_params_species():
    params = {}

    result = normalize_genex_params(params)

    assert result is params
    assert "params_species" not in result


def test_normalize_genex_params_missing_names():
    params = {
        "params_species": {
            "charge": [1, -1]
        }
    }

    result = normalize_genex_params(params)

    assert result["params_species"]["charge"] == [1, -1]


def test_normalize_genex_params_modifies_in_place():
    params = {
        "params_species": {
            "names": [" a ", " b "]
        }
    }

    result = normalize_genex_params(params)

    assert result is params
    assert params["params_species"]["names"] == ["a", "b"]


# ----------------------------------------------------------------------
# interpolate_all_sources_wrapper tests
# ----------------------------------------------------------------------

def test_interpolate_wrapper_expands_mask_and_renames_electrons(monkeypatch):
    compute = np.array([True, False, True, False])
    grid_r = np.array([1.0, 2.0, 3.0, 4.0])
    grid_z = np.array([5.0, 6.0, 7.0, 8.0])

    sub_result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    sub_result["mom1"]["ELECTRONS"]["src1"] = np.array([[10.0, 20.0]])
    sub_result["mom1"]["IONS"]["src1"] = np.array([[30.0, 40.0]])

    def fake_interpolate_all_sources(
        tria, source_dict, r_sub, z_sub,
        method="linear", fill_mode="constant", fill_value=0.0
    ):
        np.testing.assert_array_equal(r_sub, grid_r[compute])
        np.testing.assert_array_equal(z_sub, grid_z[compute])
        return sub_result

    monkeypatch.setattr(
        "genex_eirene_coupling.interpolate_all_sources",
        fake_interpolate_all_sources
    )

    out = interpolate_all_sources_wrapper(
        tria=None,
        source_dict={},
        grid_r=grid_r,
        grid_z=grid_z,
        compute=compute,
        genex_electrons="species_ELECTRONS",
    )

    np.testing.assert_array_equal(
        out["mom1"]["species_ELECTRONS"]["src1"],
        np.array([[10.0, 0.0, 20.0, 0.0]])
    )

    np.testing.assert_array_equal(
        out["mom1"]["IONS"]["src1"],
        np.array([[30.0, 0.0, 40.0, 0.0]])
    )


def test_interpolate_wrapper_keeps_electrons_name_by_default(monkeypatch):
    compute = np.array([True, True, False])

    sub_result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    sub_result["mom"]["ELECTRONS"]["src"] = np.array([[1.0, 2.0]])

    def fake_interpolate_all_sources(*args, **kwargs):
        return sub_result

    monkeypatch.setattr(
        "genex_eirene_coupling.interpolate_all_sources",
        fake_interpolate_all_sources
    )

    out = interpolate_all_sources_wrapper(
        tria=None,
        source_dict={},
        grid_r=np.array([1, 2, 3]),
        grid_z=np.array([4, 5, 6]),
        compute=compute,
    )

    assert "ELECTRONS" in out["mom"]


def test_interpolate_wrapper_preserves_dtype(monkeypatch):
    compute = np.array([False, True, True])

    sub_result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    sub_result["mom"]["IONS"]["src"] = np.array(
        [[1, 2]], dtype=np.int32
    )

    def fake_interpolate_all_sources(*args, **kwargs):
        return sub_result

    monkeypatch.setattr(
        "genex_eirene_coupling.interpolate_all_sources",
        fake_interpolate_all_sources
    )

    out = interpolate_all_sources_wrapper(
        tria=None,
        source_dict={},
        grid_r=np.array([1, 2, 3]),
        grid_z=np.array([4, 5, 6]),
        compute=compute,
    )

    assert out["mom"]["IONS"]["src"].dtype == np.int32