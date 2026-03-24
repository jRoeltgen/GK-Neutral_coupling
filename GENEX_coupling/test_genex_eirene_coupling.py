import pytest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from genex_eirene_coupling import main, status
import numpy as np

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

    tau_obj = SimpleNamespace(values=1.0)

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": SimpleNamespace(tau=tau_obj)}}
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

    tau_obj = SimpleNamespace(values=1.0)

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": SimpleNamespace(tau=tau_obj)}}
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

    tau_obj = SimpleNamespace(values=1.0)

    genex_eirene_coupling.genex_interface.load_latest_genex_fields = MagicMock(
        return_value={"es_pot": {"N/A": SimpleNamespace(tau=tau_obj)}}
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