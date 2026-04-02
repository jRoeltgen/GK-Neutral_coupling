import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from genex_eirene_coupling import (main, status, interpolate_all_moments,
                                   check_species_consistency,
                                   interpolate_all_sources)
import numpy as np
import xarray as xr
from collections import defaultdict

def make_es_pot(tau_offset):
    return xr.DataArray(
    data=np.ones((3, 4, 1895)),
    dims=("tau", "phi", "points"),
    coords={"tau": np.linspace(0, 0.005, 3) * tau_offset},
)

def test_main_single_iteration_success_DI(tmp_path):
    # ---- Args ----
    args = SimpleNamespace(
        pid=1234,
        eirene_time=10,
        MAX_TIMEOUTS=1,
        SumTemp=True,
        filepattern=str(tmp_path / "out_"),
    )

    # ---- Deps ----
    mock_deps = SimpleNamespace(
        run_eirene=MagicMock(return_value=status.SUCCESS),
        write_nc=MagicMock(),
        killpg=MagicMock(),
        sleep=MagicMock(),
        replace=MagicMock(),
        pid_exists=MagicMock(side_effect=[True, False]),
    )

    # ---- Mock EIRENE + GENEX ----
    import genex_eirene_coupling

    mock_edat = MagicMock()
    mock_edat.species_names = {"bulk_ions": ["D"]}
    mock_edat.tria = "tria"
    mock_edat.sources = {}
    mock_edat.write_ft31 = MagicMock()
    mock_edat.load_extra_forts = MagicMock()

    mock_b2dat = MagicMock()
    mock_b2dat.gmtry = "gmtry"

    genex_eirene_coupling.eirene_interface = MagicMock(return_value=(mock_edat, mock_b2dat))

    mock_grid = MagicMock()
    mock_grid.r_u = np.array([1.0])
    mock_grid.z_u = np.array([2.0])

    genex_eirene_coupling.genex_interface.initialise_genex = MagicMock(
        return_value=(mock_grid, None, None, None)
    )

    genex_eirene_coupling.genex_interface.get_genex_species = MagicMock(
        return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]
    )

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": make_es_pot(1.0)}}
    )

    genex_eirene_coupling.genex_interface.toroidal_avg = MagicMock(return_value={})

    genex_eirene_coupling.interpolate_all_moments = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_sources = MagicMock(return_value={})

    genex_eirene_coupling.prepare_fort31 = MagicMock()

    # ---- Run ----
    main(args, deps=mock_deps)

    # ---- Assertions ----
    mock_deps.run_eirene.assert_called_once()
    mock_deps.write_nc.assert_called_once()
    mock_deps.replace.assert_called_once()
    mock_deps.killpg.assert_not_called()

def test_main_sumtemp_false_raises_DI(tmp_path):
    # ---- Args ----
    args = SimpleNamespace(
        pid=1234,
        eirene_time=10,
        MAX_TIMEOUTS=1,
        SumTemp=False,
        filepattern=str(tmp_path / "out_"),
    )

    # ---- Deps ----
    mock_deps = SimpleNamespace(
        run_eirene=MagicMock(return_value=status.SUCCESS),
        write_nc=MagicMock(),
        killpg=MagicMock(),
        sleep=MagicMock(),
        replace=MagicMock(),
        pid_exists=MagicMock(side_effect=[True, False]),
    )

    # ---- Mock EIRENE + GENEX ----
    import genex_eirene_coupling

    mock_edat = MagicMock()
    mock_edat.species_names = {"bulk_ions": ["D"]}
    mock_edat.tria = "tria"
    mock_edat.sources = {}
    mock_edat.write_ft31 = MagicMock()
    mock_edat.load_extra_forts = MagicMock()

    mock_b2dat = MagicMock()
    mock_b2dat.gmtry = "gmtry"

    genex_eirene_coupling.eirene_interface = MagicMock(return_value=(mock_edat, mock_b2dat))

    mock_grid = MagicMock()
    mock_grid.r_u = np.array([1.0])
    mock_grid.z_u = np.array([2.0])

    genex_eirene_coupling.genex_interface.initialise_genex = MagicMock(
        return_value=(mock_grid, None, None, None)
    )

    genex_eirene_coupling.genex_interface.get_genex_species = MagicMock(
        return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]
    )

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": make_es_pot(1.0)}}
    )

    genex_eirene_coupling.genex_interface.toroidal_avg = MagicMock(return_value={})

    genex_eirene_coupling.interpolate_all_moments = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_sources = MagicMock(return_value={})

    genex_eirene_coupling.prepare_fort31 = MagicMock()

    # ---- Run + Assert ----
    with pytest.raises(NotImplementedError, match="SumTemp=False not implemented"):
        main(args, deps=mock_deps)

def test_main_timeout_kills_and_raises_DI():
    # ---- Args ----
    args = SimpleNamespace(
        pid=9999,
        eirene_time=10,
        MAX_TIMEOUTS=1,
        SumTemp=True,
        filepattern="out_",
    )

    # ---- Deps ----
    mock_deps = SimpleNamespace(
        run_eirene=MagicMock(return_value=status.TIMEOUT),
        write_nc=MagicMock(),
        killpg=MagicMock(),
        sleep=MagicMock(),
        replace=MagicMock(),
        pid_exists=MagicMock(side_effect=[True, False]),
    )

    # ---- Mock EIRENE + GENEX ----
    import genex_eirene_coupling

    mock_edat = MagicMock()
    mock_edat.species_names = {"bulk_ions": ["D"]}
    mock_edat.tria = "tria"
    mock_edat.sources = {}
    mock_edat.write_ft31 = MagicMock()

    mock_b2dat = MagicMock()
    genex_eirene_coupling.eirene_interface = MagicMock(return_value=(mock_edat, mock_b2dat))

    genex_eirene_coupling.genex_interface.initialise_genex = MagicMock(
        return_value=(MagicMock(r_u=np.array([1]), z_u=np.array([1])),
                      None, None, None)
    )

    genex_eirene_coupling.genex_interface.get_genex_species = MagicMock(
        return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]
    )

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": make_es_pot(1.0)}}
    )

    genex_eirene_coupling.genex_interface.toroidal_avg = MagicMock(return_value={})

    genex_eirene_coupling.interpolate_all_moments = MagicMock(return_value={})

    # ---- Run + Assert ----
    with pytest.raises(RuntimeError):
        main(args, deps=mock_deps)

    mock_deps.killpg.assert_called_once()
    mock_deps.run_eirene.assert_called_once()


def test_main_retry_and_doubling_DI(tmp_path):
    # ---- Args ----
    args = SimpleNamespace(
        pid=1234,
        eirene_time=10,
        MAX_TIMEOUTS=3,
        SumTemp=True,
        filepattern=str(tmp_path / "out_"),
    )

    # ---- Mock deps ----
    mock_run = MagicMock(side_effect=[
        status.TIMEOUT,
        status.TIMEOUT,
        status.SUCCESS,
    ])

    mock_deps = SimpleNamespace(
        run_eirene=mock_run,
        write_nc=MagicMock(),
        killpg=MagicMock(),
        sleep=MagicMock(),
        replace=MagicMock(),
        pid_exists=MagicMock(side_effect=[True, False]),
    )

    # ---- Mock external modules (minimal patching) ----
    mock_edat = MagicMock()
    mock_edat.species_names = {"bulk_ions": ["D"]}
    mock_edat.tria = "tria"
    mock_edat.sources = {}
    mock_edat.write_ft31 = MagicMock()
    mock_edat.load_extra_forts = MagicMock()

    mock_b2dat = MagicMock()
    mock_b2dat.gmtry = "gmtry"

    # Patch only unavoidable globals
    import genex_eirene_coupling

    genex_eirene_coupling.eirene_interface = MagicMock(return_value=(mock_edat, mock_b2dat))

    mock_grid = MagicMock()
    mock_grid.r_u = np.array([1.0])
    mock_grid.z_u = np.array([2.0])

    genex_eirene_coupling.genex_interface.initialise_genex = MagicMock(
        return_value=(mock_grid, None, None, None)
    )

    genex_eirene_coupling.genex_interface.get_genex_species = MagicMock(
        return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]
    )

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": make_es_pot(1.0)}}
    )

    genex_eirene_coupling.genex_interface.toroidal_avg = MagicMock(return_value={})

    genex_eirene_coupling.interpolate_all_moments = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_sources = MagicMock(return_value={})

    genex_eirene_coupling.prepare_fort31 = MagicMock()

    # ---- Run ----
    main(args, deps=mock_deps)

    # ---- Assertions ----
    assert mock_run.call_count == 3

    timeouts_used = [call.args[0] for call in mock_run.call_args_list]
    assert timeouts_used == [10, 20, 40]

    mock_deps.write_nc.assert_called_once()
    mock_deps.replace.assert_called_once()

def test_main_multiple_iterations(tmp_path):
    # ---- Args ----
    args = SimpleNamespace(
        pid=1234,
        eirene_time=10,
        MAX_TIMEOUTS=2,
        SumTemp=True,
        filepattern=str(tmp_path / "out_"),
    )

    # ---- Mock dependencies ----
    # EIRENE status per call: success for all calls
    mock_run_eirene = MagicMock(side_effect=[status.SUCCESS] * 3)
    mock_write_nc = MagicMock()
    mock_killpg = MagicMock()
    mock_sleep = MagicMock()
    mock_replace = MagicMock()
    # pid_exists: 3 True then False to exit loop
    mock_pid_exists = MagicMock(side_effect=[True, True, True, False])

    deps = SimpleNamespace(
        run_eirene=mock_run_eirene,
        write_nc=mock_write_nc,
        killpg=mock_killpg,
        sleep=mock_sleep,
        replace=mock_replace,
        pid_exists=mock_pid_exists,
    )

    # ---- Mock EIRENE + GENEX objects ----
    mock_edat = MagicMock()
    mock_edat.species_names = {"bulk_ions": ["D"]}
    mock_edat.tria = "tria"
    mock_edat.sources = {"D": np.array([1.0])}
    mock_edat.write_ft31 = MagicMock()
    mock_edat.load_extra_forts = MagicMock()

    mock_b2dat = MagicMock()
    mock_b2dat.gmtry = "gmtry"

    import genex_eirene_coupling

    genex_eirene_coupling.eirene_interface = MagicMock(
        return_value=(mock_edat, mock_b2dat)
    )

    mock_grid = MagicMock()
    mock_grid.r_u = np.array([1.0])
    mock_grid.z_u = np.array([2.0])

    genex_eirene_coupling.genex_interface.initialise_genex = MagicMock(
        return_value=(mock_grid, None, None, None)
    )

    genex_eirene_coupling.genex_interface.get_genex_species = MagicMock(
        return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]
    )

    # Provide increasing tau values to pass last_tau check
    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        side_effect=[{"es_pot": {"N/A": make_es_pot(1.0)}},
                     {"es_pot": {"N/A": make_es_pot(2.0)}},
                     {"es_pot": {"N/A": make_es_pot(3.0)}},
        ]
    )

    genex_eirene_coupling.genex_interface.toroidal_avg = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_moments = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_sources = MagicMock(return_value={})

    # ---- Run ----
    main(args, deps=deps)

    # ---- Assertions ----
    # Loop should have run 3 iterations
    assert mock_run_eirene.call_count == 3
    assert mock_write_nc.call_count == 3
    # replace should be called for each iteration
    assert mock_replace.call_count == 3
    # pid_exists should have been called at least 4 times
    assert mock_pid_exists.call_count >= 4
    # killpg should not be called (no timeout exceeded)
    mock_killpg.assert_not_called()
    # sleep should not be called as tau always increases
    mock_sleep.assert_not_called()


def test_main_last_tau_prevents_iteration(tmp_path):

    # ---- Args ----
    args = SimpleNamespace(
        pid=1234,
        eirene_time=10,
        MAX_TIMEOUTS=2,
        SumTemp=True,
        filepattern=str(tmp_path / "out_"),
    )

    # ---- Dependencies ----
    mock_run_eirene = MagicMock(return_value=status.SUCCESS)
    mock_write_nc = MagicMock()
    mock_killpg = MagicMock()
    mock_sleep = MagicMock()
    mock_replace = MagicMock()

    # pid_exists: 3 True then False to exit loop
    mock_pid_exists = MagicMock(side_effect=[True, True, True, False])

    deps = SimpleNamespace(
        run_eirene=mock_run_eirene,
        write_nc=mock_write_nc,
        killpg=mock_killpg,
        sleep=mock_sleep,
        replace=mock_replace,
        pid_exists=mock_pid_exists,
    )

    # ---- Mock EIRENE/GENEX ----
    import genex_eirene_coupling

    mock_edat = MagicMock()
    mock_edat.species_names = {"bulk_ions": ["D"]}
    mock_edat.tria = "tria"
    mock_edat.sources = {"D": np.array([1.0])}
    mock_edat.write_ft31 = MagicMock()
    mock_edat.load_extra_forts = MagicMock()

    mock_b2dat = MagicMock()
    mock_b2dat.gmtry = "gmtry"

    genex_eirene_coupling.eirene_interface = MagicMock(
        return_value=(mock_edat, mock_b2dat)
    )

    mock_grid = MagicMock()
    mock_grid.r_u = np.array([1.0])
    mock_grid.z_u = np.array([2.0])

    genex_eirene_coupling.genex_interface.initialise_genex = MagicMock(
        return_value=(mock_grid, None, None, None)
    )

    genex_eirene_coupling.genex_interface.get_genex_species = MagicMock(
        return_value=[
            SimpleNamespace(name="D", is_electron=False),
            SimpleNamespace(name="e", is_electron=True),
        ]
    )

    # ---- FORCE SAME tau twice ----
    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        side_effect=[{"es_pot": {"N/A": make_es_pot(1.0)}},
                {"es_pot": {"N/A": make_es_pot(1.0)}},
                {"es_pot": {"N/A": make_es_pot(2.0)}},
        ]
    )

    genex_eirene_coupling.genex_interface.toroidal_avg = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_moments = MagicMock(return_value={})
    genex_eirene_coupling.interpolate_all_sources = MagicMock(return_value={})

    # ---- Run ----
    main(args, deps=deps)

    # ---- Assertions ----

    # Only TWO iterations should complete (second is skipped)
    assert mock_run_eirene.call_count == 2

    # Two output write
    assert mock_write_nc.call_count == 2
    assert mock_replace.call_count == 2

    # sleep should be triggered due to repeated tau
    mock_sleep.assert_called()

    # second loop iteration was skipped → no second EIRENE run
    assert mock_pid_exists.call_count >= 3

    # no failure path
    mock_killpg.assert_not_called()

def test_interpolate_all_moments_indices(monkeypatch):
    # --- Mock interp_moments to record calls ---
    calls = []
    def fake_interp(gmtry, grid_r, grid_z, value, ind):
        calls.append(ind)
        return {"mocked": True}

    monkeypatch.setattr("genex_eirene_coupling.interp_moments", fake_interp)

    gmtry = "gmtry"
    grid_r = np.array([1.0])
    grid_z = np.array([2.0])
    genex_out = {
        "poloidal_fluxes": {"D": np.ones((3, 3))},
        "radial_fluxes": {"D": np.ones((3, 3))},
        "other_field": {"D": np.ones((3, 3))},
    }

    result = interpolate_all_moments(gmtry, grid_r, grid_z, genex_out)

    # --- Check that correct indices are selected per field ---
    assert calls[0] == [0, 2]  # poloidal_fluxes
    assert calls[1] == [2, 3]  # radial_fluxes
    assert calls[2] == [0, 1, 2, 3]  # other_field

    # --- Check output is nested defaultdict ---
    assert isinstance(result, defaultdict)
    assert isinstance(result["poloidal_fluxes"], defaultdict)
    assert result["poloidal_fluxes"]["D"] == {"mocked": True}

def test_check_species_consistency_raises():
    eirene_species = ["D"]
    genex_species = [
        SimpleNamespace(name="T", is_electron=False),
        SimpleNamespace(name="e", is_electron=True),
    ]

    with pytest.raises(ValueError) as exc:
        check_species_consistency(eirene_species, genex_species)
    assert "Genex Species T not known to Eirene" in str(exc.value)

def test_get_genex_electron_name_returns_correct_name():
    from genex_eirene_coupling import get_genex_electron_name
    genex_species = [
        SimpleNamespace(name="D", is_electron=False),
        SimpleNamespace(name="e", is_electron=True),
        SimpleNamespace(name="T", is_electron=False),
    ]
    electron_name = get_genex_electron_name(genex_species)
    assert electron_name == "e"
