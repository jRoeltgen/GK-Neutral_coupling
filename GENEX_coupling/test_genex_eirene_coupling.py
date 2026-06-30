import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from genex_eirene_coupling import (main, status, interpolate_all_moments,
                                   check_species_consistency,
                                   interpolate_all_sources_wrapper,
                                   interpolate_temperature_values,
                                   normalize_genex_params,
                                   backup_eirene_files,
                                   next_eirene_index)
import numpy as np
import xarray as xr
from collections import defaultdict
from pathlib import Path

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
        filepattern="out",
        genex_time_index_override=False,
        eirene_path=tmp_path,
        genex_path=tmp_path,
        eirene_command="eirobjx",
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
        collision_mappers=None,
        sparse_temperature_handler=mod.default_sparse_temperature_handler,
        pseudo_temperature_handler=(
            mod.default_single_species_pseudo_temperature_handler
        ),
    )

    # ----------------------------
    # Grid (must support .values)
    # ----------------------------
    grid = MagicMock()
    grid.r_u = xr.DataArray([1.0])
    grid.z_u = xr.DataArray([2.0])

    r_all = grid.r_u
    z_all = grid.z_u
    compute = np.array([True])

    norm = {"R0": 1.0}

    params = {
        "params_species": {},
        "params_time_loop": {"start_from_checkpoint": False},
    }

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
    edat.sources = {
        "particle": {
            "D": {"SUM": np.array([1.0]), "stratum_1": np.array([2.0])},
            "ELECTRONS": {"SUM": np.array([3.0])},
        }
    }
    (tmp_path / "fort.31").write_text("original fort.31")

    def write_ft31(path):
        Path(path).write_text("updated fort.31")

    edat.write_ft31 = MagicMock(side_effect=write_ft31)
    edat.load_extra_forts = MagicMock()

    b2dat = MagicMock()
    b2dat.gmtry = "gmtry"

    monkeypatch.setattr(
        mod,
        "eirene_interface",
        MagicMock(return_value=(
            edat,
            b2dat,
            np.array([False]),
            np.array([False]),
        )),
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
    assert env["deps"].run_eirene.call_args.kwargs["command"] == "eirobjx"
    env["deps"].write_nc.assert_called_once()
    env["deps"].replace.assert_called_once()
    env["deps"].killpg.assert_not_called()
    assert (env["args"].eirene_path / "eirene_sources_000000" / "fort.31").exists()

def test_main_timeout_kills_and_raises(coupling_env):
    env = coupling_env

    env["deps"].run_eirene.return_value = env["mod"].status.TIMEOUT

    with pytest.raises(RuntimeError):
        env["mod"].main(env["args"], deps=env["deps"])

    env["deps"].killpg.assert_called_once()

def test_main_eirene_error_kills_and_raises(coupling_env):
    env = coupling_env

    env["deps"].run_eirene.return_value = env["mod"].status.ERROR

    with pytest.raises(RuntimeError, match="EIRENE failed"):
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
    assert (env["args"].eirene_path / "eirene_sources_000002" / "fort.31").exists()

def test_main_retries_transient_hdf_error(coupling_env):
    env = coupling_env

    env["mod"].genex_interface.load_latest_genex_fields.side_effect = [
        RuntimeError("HDF error while reading"),
        (env["fields"], 0.001),
    ]

    env["mod"].main(env["args"], deps=env["deps"])

    assert env["mod"].genex_interface.load_latest_genex_fields.call_count == 2
    env["deps"].run_eirene.assert_called_once()

def test_main_repeated_hdf_errors_raise(coupling_env):
    env = coupling_env

    env["mod"].genex_interface.load_latest_genex_fields.side_effect = [
        RuntimeError("HDF error while reading"),
        RuntimeError("HDF error while reading"),
        RuntimeError("HDF error while reading"),
    ]

    with pytest.raises(RuntimeError, match="Repeated NetCDF HDF errors"):
        env["mod"].main(env["args"], deps=env["deps"])

    env["deps"].run_eirene.assert_not_called()

def test_main_checkpoint_uses_next_eirene_index(coupling_env):
    env = coupling_env
    env["mod"].genex_interface.wait_for_genex_init.return_value[2][
        "params_time_loop"
    ]["start_from_checkpoint"] = True
    (env["args"].eirene_path / "out_000003.nc").touch()

    env["mod"].main(env["args"], deps=env["deps"])

    assert (env["args"].eirene_path / "eirene_sources_000004" / "fort.31").exists()
    assert env["deps"].write_nc.call_args.args[0].endswith("out_000004.nc.tmp")

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

def test_main_sumtemp_false_writes_temperatures(coupling_env, monkeypatch):
    env = coupling_env
    env["args"].SumTemp = False
    env["mod"].genex_interface.toroidal_avg.return_value = {
        "Ttot": {"D": np.array([4.0])}
    }
    env["edat"].full_source_in_SI = {}
    monkeypatch.setattr(
        env["mod"],
        "get_temperatures",
        MagicMock(return_value={"Ti_D": np.array([1.0])}),
    )
    fake_processor = MagicMock()
    fake_processor.regroup_by_temperature.return_value = (
        {},
        {"Ti_D": np.array([1.0])},
    )
    monkeypatch.setattr(
        env["mod"].SPP,
        "SourcePostProcessor",
        MagicMock(return_value=fake_processor),
    )

    custom_mappers = {"atom-plasma": MagicMock()}
    custom_sparse_handler = MagicMock()
    custom_pseudo_handler = MagicMock()
    env["deps"].collision_mappers = custom_mappers
    env["deps"].sparse_temperature_handler = custom_sparse_handler
    env["deps"].pseudo_temperature_handler = custom_pseudo_handler
    env["mod"].main(env["args"], deps=env["deps"])

    kwargs = env["deps"].write_nc.call_args.kwargs
    assert kwargs["write_temperature"] is True
    np.testing.assert_allclose(
        kwargs["temperature_values"]["Ti_D"],
        np.array([[4.0 * 1.602176634e-19]]),
    )
    assert (
        env["mod"].SPP.SourcePostProcessor.call_args.kwargs[
            "collision_mappers"
        ]
        is custom_mappers
    )
    assert (
        env["mod"].get_temperatures.call_args.kwargs[
            "sparse_temperature_handler"
        ]
        is custom_sparse_handler
    )
    assert (
        env["mod"].get_temperatures.call_args.kwargs[
            "pseudo_temperature_handler"
        ]
        is custom_pseudo_handler
    )

def test_main_filters_sources_to_sum_only(coupling_env):
    env = coupling_env

    env["mod"].main(env["args"], deps=env["deps"])

    source_arg = env["mod"].interpolate_all_sources.call_args.args[1]
    assert set(source_arg) == {"particle"}
    assert set(source_arg["particle"]) == {"D", "ELECTRONS"}
    assert set(source_arg["particle"]["D"]) == {"SUM"}
    assert set(source_arg["particle"]["ELECTRONS"]) == {"SUM"}
    np.testing.assert_array_equal(source_arg["particle"]["D"]["SUM"], np.array([1.0]))
    np.testing.assert_array_equal(
        source_arg["particle"]["ELECTRONS"]["SUM"], np.array([3.0])
    )

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
        return np.arange(4.0)

    monkeypatch.setattr("genex_eirene_coupling.interp_moments", fake_interp)

    class FakeValue:
        def __init__(self, arr):
            self.data = arr

    gmtry = "gmtry"
    tri = "tri"

    genex_out = {
        "fnax": {"D": FakeValue(np.ones((3, 3)))},
        "fnay": {"D": FakeValue(np.ones((3, 3)))},
        "other": {"D": FakeValue(np.ones((3, 3)))},
    }

    poloidal_mask = np.array([False, True, False, False])
    radial_mask = np.array([False, False, True, False])

    result = interpolate_all_moments(
        gmtry, tri, genex_out, radial_mask, poloidal_mask
    )

    assert calls[0] == [0, 2]
    assert calls[1] == [2, 3]
    assert calls[2] == [0, 1, 2, 3]

    np.testing.assert_array_equal(result["fnax"]["D"], np.array([0.0, 0.0, 2.0, 3.0]))
    np.testing.assert_array_equal(result["fnay"]["D"], np.array([0.0, 1.0, 0.0, 3.0]))
    np.testing.assert_array_equal(result["other"]["D"], np.arange(4.0))


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

def test_backup_eirene_files_copies_fort31_and_moves_matching_files(tmp_path):
    (tmp_path / "fort.31").write_text("fort31")
    (tmp_path / "fort.44").write_text("fort44")
    (tmp_path / "fort.12").write_text("fort12")
    (tmp_path / "fort.123").write_text("fort123")
    (tmp_path / "fort.9").write_text("keep")
    (tmp_path / "notes.txt").write_text("keep")

    backup_eirene_files(tmp_path, 7)

    dest = tmp_path / "eirene_sources_000007"
    assert dest.is_dir()
    assert (dest / "fort.31").read_text() == "fort31"
    assert (dest / "fort.44").read_text() == "fort44"
    assert (dest / "fort.12").read_text() == "fort12"
    assert (dest / "fort.123").read_text() == "fort123"
    assert (tmp_path / "fort.31").exists()
    assert not (tmp_path / "fort.44").exists()
    assert not (tmp_path / "fort.12").exists()
    assert not (tmp_path / "fort.123").exists()
    assert (tmp_path / "fort.9").exists()
    assert (tmp_path / "notes.txt").exists()

def test_backup_eirene_files_ignores_matching_directories(tmp_path):
    (tmp_path / "fort.31").write_text("fort31")
    (tmp_path / "fort.123").mkdir()

    backup_eirene_files(tmp_path, 0)

    assert (tmp_path / "fort.123").is_dir()
    assert not (tmp_path / "eirene_sources_000000" / "fort.123").exists()

def test_next_eirene_index_returns_next_after_highest_match(tmp_path):
    (tmp_path / "input_sources_000001.nc").touch()
    (tmp_path / "input_sources_000007.nc").touch()
    (tmp_path / "input_sources_bad.nc").touch()
    (tmp_path / "other_000009.nc").touch()

    assert next_eirene_index(tmp_path, "input_sources") == 8

def test_next_eirene_index_returns_zero_without_matches(tmp_path):
    assert next_eirene_index(tmp_path, "input_sources") == 0


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


def test_temperature_interpolation_reuses_direct_values(monkeypatch):
    compute = np.array([True, False, True])
    interpolate = MagicMock(return_value=np.array([10.0, 20.0]))
    monkeypatch.setattr(
        "genex_eirene_coupling.interpolate_source", interpolate
    )

    out = interpolate_temperature_values(
        tria="mesh",
        temperature_values={
            "Tn_D": np.array([1.0, 2.0]),
            "Ti_D": np.array([3.0, 4.0]),
        },
        grid_r=np.array([0.0, 1.0, 2.0]),
        grid_z=np.array([0.0, 1.0, 2.0]),
        compute=compute,
        direct_values={"Ti_D": np.array([30.0, 40.0])},
    )

    np.testing.assert_array_equal(out["Tn_D"], [[10.0, 0.0, 20.0]])
    np.testing.assert_array_equal(out["Ti_D"], [[30.0, 0.0, 40.0]])
    interpolate.assert_called_once()
