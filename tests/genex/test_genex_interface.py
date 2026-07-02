import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from torx.normalization.normalization_m import Normalization
from pint import UnitRegistry
from types import SimpleNamespace
from pathlib import Path
ureg = UnitRegistry()

from neutral_coupling.genex_coupling.genex_interface import (
    wait_for_genex_init,
    get_grid_with_ghost_filler,
    load_latest_genex_fields,
    calculate_temperatures,
    toroidal_avg,
    species,
    get_genex_species,
    retry_compute,
)

@pytest.fixture
def fake_norm():
    return Normalization({
        "Te0": 0.04 * ureg.keV,
        "Ti0": 0.04 * ureg.keV,
        "n0": 1.0e19 / (ureg.meter**(3)),
        "c_s0": 1.0 * ureg.meter / ureg.second,
        "elementary_charge": 1.0 * ureg.e,
        "Mi": 2.0 * ureg.u,
        "R0": 1.0 * ureg.meter,
    })


@pytest.fixture
def fake_grid():
    grid = MagicMock()

    grid.vector_to_matrix.side_effect = lambda x: x
    grid.matrix_to_vector.side_effect = lambda x: x

    return grid


@pytest.fixture
def fake_data():
    return xr.DataArray(
        np.ones((3, 4)),
        dims=("RZ", "phi"),
        coords={"RZ": np.arange(3), "phi": np.arange(4)},
    )

@pytest.fixture
def fake_species():
    return [
        species("e", -1),   # electron
        species("D", +1),   # ion
    ]

@pytest.fixture(autouse=True)
def clear_load_latest_ref_mask():
    if hasattr(load_latest_genex_fields, "ref_mask"):
        delattr(load_latest_genex_fields, "ref_mask")
    yield
    if hasattr(load_latest_genex_fields, "ref_mask"):
        delattr(load_latest_genex_fields, "ref_mask")

@pytest.fixture
def mocked_io():
    with patch("neutral_coupling.genex_coupling.genex_interface.filepath_resolver") as mock_resolver, \
         patch("neutral_coupling.genex_coupling.genex_interface.xr.open_dataset") as mock_open:

        mock_resolver.return_value = Path("mesh.nc")
        yield mock_open

@patch("neutral_coupling.genex_coupling.genex_interface.get_grid_with_ghost_filler")
@patch("neutral_coupling.genex_coupling.genex_interface.initialize_genex_from_filepath")
def test_wait_for_genex_init_immediate_success(mock_init, mock_grid_loader):
    mock_grid = SimpleNamespace(r_u=1, z_u=1)
    mock_init.return_value = (mock_grid, "equi", "params", "norm")
    mock_grid_loader.return_value = ("r_all", "z_all", "compute_mask")

    result = wait_for_genex_init("path", timeout=1, poll=0)

    assert result == (mock_grid, "equi", "params", "norm", "r_all",
                      "z_all", "compute_mask")
    mock_init.assert_called_once_with("path")

@patch("neutral_coupling.genex_coupling.genex_interface.get_grid_with_ghost_filler")
@patch("neutral_coupling.genex_coupling.genex_interface.initialize_genex_from_filepath")
def test_wait_for_genex_init_retries_until_valid(mock_init, mock_grid_loader):
    calls = {"n": 0}

    def fake_loader(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("not ready")

        mock_grid = SimpleNamespace(r_u=1, z_u=1)
        return mock_grid, "equi", "params", "norm"

    mock_init.side_effect = fake_loader
    mock_grid_loader.return_value = ("r_all", "z_all", "compute_mask")

    result = wait_for_genex_init("dummy_path", timeout=1, poll=0)

    assert result[0].r_u == 1
    assert calls["n"] == 3

@pytest.mark.parametrize(
    "coord_names,phi_dim,rvals,zvals,ghost,filler,expected",
    [
        (
            ("R", "Z"),
            "dim_phi",
            [1,2,3],
            [4,5,6],
            [1,0,1],
            [1,1,0],
            [True,False,False],
        ),
        (
            ("x", "y"),
            "dim_phi",
            [10,20],
            [30,40],
            [1,1],
            [1,0],
            [True,False],
        ),
        (
            ("R", "Z"),
            "dim_RZ_grid_phi",
            [7,8],
            [9,10],
            [1,1],
            [1,1],
            [True,True],
        ),
    ]
)
def test_get_grid_valid_cases(
    mocked_io,
    coord_names,
    phi_dim,
    rvals,
    zvals,
    ghost,
    filler,
    expected,
):
    rname, zname = coord_names

    ds = xr.Dataset({
        rname: ((phi_dim, "n"), [rvals]),
        zname: ((phi_dim, "n"), [zvals]),
        "not_ghost": ((phi_dim, "n"), [ghost]),
        "not_filler": ((phi_dim, "n"), [filler]),
    })

    mocked_io.return_value = ds

    r, z, compute = get_grid_with_ghost_filler(Path("dummy"))

    assert (r.values == rvals).all()
    assert (z.values == zvals).all()
    assert (compute.values == expected).all()

def test_get_grid_unknown_dimension(mocked_io):
    ds = xr.Dataset({
        "R": (("foo", "n"), [[1,2]]),
        "Z": (("foo", "n"), [[3,4]]),
        "not_ghost": (("foo", "n"), [[1,1]]),
        "not_filler": (("foo", "n"), [[1,1]]),
    })

    mocked_io.return_value = ds

    with pytest.raises(NotImplementedError):
        get_grid_with_ghost_filler(Path("dummy"))

def test_get_grid_missing_mask(mocked_io):
    ds = xr.Dataset({
        "R": (("dim_phi", "n"), [[1]]),
        "Z": (("dim_phi", "n"), [[2]]),
        "not_filler": (("dim_phi", "n"), [[1]]),
    })

    mocked_io.return_value = ds

    with pytest.raises(KeyError):
        get_grid_with_ghost_filler(Path("dummy"))

@patch("neutral_coupling.genex_coupling.genex_interface.parallel_temperature")
@patch("neutral_coupling.genex_coupling.genex_interface.perpendicular_temperature")
def test_calculate_temperatures(mock_perp, mock_par, fake_norm, fake_data):

    mock_par.return_value = fake_data.copy()
    mock_par.return_value.attrs["norm"] = 123

    mock_perp.return_value = fake_data.copy()

    Ttot, Tpar, Tperp = calculate_temperatures(
        params="params",
        norm=fake_norm,
        spec="D",
        n=fake_data.copy(),
        u_par=fake_data.copy(),
        E_par=fake_data.copy(),
        E_perp=fake_data.copy(),
    )

    assert np.allclose(Ttot, (Tpar + 2*Tperp)/3)
    assert Ttot.attrs["norm"] == 123

    mock_par.assert_called_once()
    mock_perp.assert_called_once()

def test_load_latest_genex_fields(
    fake_grid,
    fake_norm,
    fake_data,
    fake_species,
):
    with (
        patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex") as mock_load,
        patch("neutral_coupling.genex_coupling.genex_interface.electric_field") as mock_efield,
        patch("neutral_coupling.genex_coupling.genex_interface.velocities_m") as mock_vel,
        patch("neutral_coupling.genex_coupling.genex_interface.electrostatic_ExB_heat_flux") as mock_q,
        patch("neutral_coupling.genex_coupling.genex_interface.calculate_temperatures") as mock_calc_temp,
        patch("neutral_coupling.genex_coupling.genex_interface.total_pressure") as mock_total_pressure,
        patch("neutral_coupling.genex_coupling.genex_interface.wait_until_genex_stable") as mock_wait,
    ):

        mock_wait.return_value = np.array([0.0, 0.001])

        # ---- mock data loading ----
        def fake_loader(path, spec, field):
            da = fake_data.copy()

            # emulate "already time-indexed" output
            da = da.expand_dims(tau=[0.0, 0.001])

            return da
        mock_load.side_effect = fake_loader

        # ---- mock derived functions ----
        mock_efield.return_value = "efield"
        mock_vel.ExB_velocity.return_value = 1.0
        mock_vel.diamagnetic_velocity.return_value = 2.0
        mock_total_pressure.return_value = MagicMock(values=fake_data)
        uvec = xr.DataArray(
            np.ones((3, 3, 4)),  # (vector, RZ, phi)
            dims=("vector", "RZ", "phi"),
            coords={
                "vector": ["eR", "ePhi", "eZ"],
                "RZ": [0, 1, 2],
                "phi": [0, 1, 2, 3],
            }
        )
        mock_vel.parallel_ion_velocity_vector.return_value = uvec

        mock_q.return_value = fake_data

        mock_calc_temp.return_value = (fake_data, None, None)

        out, time = load_latest_genex_fields(
            gpath=Path("path"),
            all_spec=fake_species,
            grid=fake_grid,
            equi="equi",
            params="params",
            norm=fake_norm,
            time_index=-1,
            timeout=1,
        )

        assert time == mock_wait.return_value[-1]
        mock_wait.assert_called_once()

        # ---- structure checks ----
        assert "n" in out
        assert "u_par" in out
        assert "Ttot" in out
        assert "u_rad" in out
        assert "pr" in out

        spec = []
        for sp in fake_species:
            spec.append(sp.name)
        for s in spec:
            assert s in out["n"]
            assert s in out["Ttot"]
            assert s in out["u_rad"]

        # es_pot and pr special cases
        assert "N/A" in out["es_pot"]
        assert "N/A" in out["pr"]
        assert len(out["pr"]) == 1

        # radial velocity combination
        for s in spec:
            assert out["u_rad"][s] == 1.0 + 2.0

        # ensure temperature calculation called
        assert mock_calc_temp.call_count == len(spec)

@patch("neutral_coupling.genex_coupling.genex_interface.wait_until_genex_stable")
@patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex")
def test_load_latest_genex_fields_time_index_too_large(
    mock_load, mock_wait, fake_grid, fake_norm, fake_species
):
    mock_wait.return_value = np.array([0.0, 1.0])

    # Safety: ensure no data loading happens
    mock_load.side_effect = AssertionError("Should not load data")

    with pytest.raises(ValueError, match="time_index"):
        load_latest_genex_fields(
            gpath="path",
            all_spec=fake_species,
            grid=fake_grid,
            equi="equi",
            params="params",
            norm=fake_norm,
            time_index=5,
            timeout=1,
        )

def test_load_latest_genex_fields_uses_requested_stable_time_index(
    fake_grid,
    fake_norm,
    fake_data,
    fake_species,
):
    with (
        patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex") as mock_load,
        patch("neutral_coupling.genex_coupling.genex_interface.electric_field") as mock_efield,
        patch("neutral_coupling.genex_coupling.genex_interface.velocities_m") as mock_vel,
        patch("neutral_coupling.genex_coupling.genex_interface.electrostatic_ExB_heat_flux") as mock_q,
        patch("neutral_coupling.genex_coupling.genex_interface.calculate_temperatures") as mock_calc_temp,
        patch("neutral_coupling.genex_coupling.genex_interface.total_pressure") as mock_total_pressure,
        patch("neutral_coupling.genex_coupling.genex_interface.wait_until_genex_stable") as mock_wait,
    ):
        tau = np.array([0.0, 0.001, 0.002])
        mock_wait.return_value = tau

        def fake_loader(path, spec, field):
            return fake_data.copy().expand_dims(tau=tau)

        mock_load.side_effect = fake_loader
        mock_efield.return_value = "efield"
        mock_vel.ExB_velocity.return_value = 1.0
        mock_vel.diamagnetic_velocity.return_value = 2.0
        mock_total_pressure.return_value = MagicMock(values=fake_data)
        mock_vel.parallel_ion_velocity_vector.return_value = xr.DataArray(
            np.ones((3, 3, 4)),
            dims=("vector", "RZ", "phi"),
            coords={
                "vector": ["eR", "ePhi", "eZ"],
                "RZ": [0, 1, 2],
                "phi": [0, 1, 2, 3],
            },
        )
        mock_q.return_value = fake_data
        mock_calc_temp.return_value = (fake_data, None, None)

        out, time = load_latest_genex_fields(
            gpath=Path("path"),
            all_spec=fake_species,
            grid=fake_grid,
            equi="equi",
            params="params",
            norm=fake_norm,
            time_index=1,
            timeout=1,
        )

        assert time == tau[1]
        assert set(out["n"]) == {"e", "D"}

@patch("neutral_coupling.genex_coupling.genex_interface._diagnostic_print")
@patch("neutral_coupling.genex_coupling.genex_interface.wait_until_genex_stable")
@patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex")
def test_load_latest_genex_fields_zero_mask_change_raises(
    mock_load,
    mock_wait,
    mock_diagnostic,
    fake_grid,
    fake_norm,
    fake_species,
):
    tau = np.array([0.0, 1.0])
    load_latest_genex_fields.ref_mask = np.zeros((3, 4), dtype=bool)
    mock_wait.return_value = tau

    def fake_loader(path, spec, field):
        data = xr.DataArray(
            np.ones((3, 4)),
            dims=("RZ", "phi"),
            coords={"RZ": np.arange(3), "phi": np.arange(4)},
        )
        if field == "n":
            data = data.copy()
            data.values[0, 0] = 0.0
        return data.expand_dims(tau=tau)

    mock_load.side_effect = fake_loader

    with (
        patch("neutral_coupling.genex_coupling.genex_interface.electric_field", return_value="efield"),
        patch("neutral_coupling.genex_coupling.genex_interface.velocities_m") as mock_vel,
    ):
        mock_vel.ExB_velocity.return_value = 1.0

        with pytest.raises(ValueError, match="Zero mask changed"):
            load_latest_genex_fields(
                gpath=Path("path"),
                all_spec=fake_species,
                grid=fake_grid,
                equi="equi",
                params="params",
                norm=fake_norm,
                time_index=-1,
                timeout=1,
            )

    mock_diagnostic.assert_called_once()

def test_retry_compute_returns_immediately_on_success():
    fn = MagicMock(return_value="done")

    assert retry_compute(fn, attempts=3, delay=0) == "done"
    fn.assert_called_once_with()

@patch("neutral_coupling.genex_coupling.genex_interface.time.sleep")
def test_retry_compute_retries_hdf_errors(mock_sleep):
    fn = MagicMock(side_effect=[RuntimeError("HDF error while reading"), "done"])

    assert retry_compute(fn, attempts=3, delay=0.25) == "done"
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(0.25)

@patch("neutral_coupling.genex_coupling.genex_interface.time.sleep")
def test_retry_compute_reraises_non_hdf_errors(mock_sleep):
    fn = MagicMock(side_effect=RuntimeError("different failure"))

    with pytest.raises(RuntimeError, match="different failure"):
        retry_compute(fn, attempts=3, delay=0.25)

    fn.assert_called_once_with()
    mock_sleep.assert_not_called()

@patch("neutral_coupling.genex_coupling.genex_interface.time.sleep")
def test_retry_compute_raises_after_repeated_hdf_errors(mock_sleep):
    fn = MagicMock(side_effect=RuntimeError("HDF error while reading"))

    with pytest.raises(RuntimeError, match="Repeated HDF errors"):
        retry_compute(fn, attempts=3, delay=0.25)

    assert fn.call_count == 3
    assert mock_sleep.call_count == 3

def test_toroidal_avg(fake_data):

    genex_out = {
        "density": {
            "D": fake_data,
            "C": fake_data * 2,
        }
    }

    out = toroidal_avg(genex_out)

    for s in ["D", "C"]:
        expected = genex_out["density"][s].mean(dim="phi")
        assert np.allclose(out["density"][s], expected)

def test_multiple_ions_raises(fake_grid, fake_norm):
    all_spec = [
        species("e", -1),
        species("D", +1),
        species("C", +6),
    ]

    with pytest.raises(ValueError, match="2\\+ ion"):
        load_latest_genex_fields(
            gpath="path",
            all_spec=all_spec,
            grid=fake_grid,
            equi="equi",
            params="params",
            norm=fake_norm,
            time_index=-1,
            timeout=1,
        )

def test_multiple_electrons_raises(fake_grid, fake_norm):
    all_spec = [
        species("e1", -1),
        species("e2", -1),
        species("D", +1),
    ]

    with pytest.raises(ValueError, match="Multiple assumed electrons"):
        load_latest_genex_fields(
            gpath="path",
            all_spec=all_spec,
            grid=fake_grid,
            equi="equi",
            params="params",
            norm=fake_norm,
            time_index=-1,
            timeout=1,
        )

def test_get_genex_species_basic():
    params = {
        "params_species": {
            "names": ["e", "D"],
            "charges": [-1, 1],
        }
    }

    result = get_genex_species(params)

    assert len(result) == 2
    assert result[0].name == params["params_species"]["names"][0]
    assert result[0].charge == params["params_species"]["charges"][0]
    assert result[0].is_electron is True
    assert result[1].name == "D"
    assert result[1].charge == 1
    assert result[1].is_electron is False

def test_get_genex_species_ignores_empty():
    params = {
        "params_species": {
            "names": ["e", " ", "", "D"],
            "charges": [-1, 0, 0, 1],
        }
    }

    result = get_genex_species(params)

    assert len(result) == 2
    names = [s.name for s in result]
    assert names == ["e", "D"]

def test_get_genex_species_charge_alignment():
    params = {
        "params_species": {
            "names": ["e", "D"],
            "charges": [-2, 3],  # unusual but valid
        }
    }

    result = get_genex_species(params)

    assert result[0].charge == -2
    assert result[1].charge == 3

def test_get_mom_paths_partitioned(tmp_path):
    # create part directories
    p1 = tmp_path / "part_0"
    p1.mkdir()
    (p1 / "mom_2d.nc").touch()

    p2 = tmp_path / "part_1"
    p2.mkdir()
    (p2 / "mom_2d.nc").touch()

    from neutral_coupling.genex_coupling.genex_interface import get_mom_paths

    result = get_mom_paths(tmp_path)

    assert len(result) == 2
    assert all(p.name == "mom_2d.nc" for p in result)


def test_get_mom_paths_single_file(tmp_path):
    f = tmp_path / "mom_2d.nc"
    f.touch()

    from neutral_coupling.genex_coupling.genex_interface import get_mom_paths

    result = get_mom_paths(tmp_path)

    assert result == [f]


def test_get_mom_paths_none(tmp_path):
    from neutral_coupling.genex_coupling.genex_interface import get_mom_paths

    result = get_mom_paths(tmp_path)

    assert result == []

@patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex")
def test_get_tau_for_vars_consistent(mock_load):
    import xarray as xr
    tau = np.array([0.0, 1.0, 2.0])

    def fake_loader(path, spec, var):
        return xr.DataArray(
            np.zeros((3,)),
            dims=("tau",),
            coords={"tau": tau},
        )

    mock_load.side_effect = fake_loader

    from neutral_coupling.genex_coupling.genex_interface import get_tau_for_vars

    result = get_tau_for_vars("path", ["e", "D"])

    assert np.array_equal(result, tau)


@patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex")
def test_get_tau_for_vars_inconsistent(mock_load):
    import xarray as xr

    def fake_loader(path, spec, var):
        if spec == "e":
            tau = np.array([0.0, 1.0])
        else:
            tau = np.array([0.0, 2.0])
        return xr.DataArray(
            np.zeros((len(tau),)),
            dims=("tau",),
            coords={"tau": tau},
        )

    mock_load.side_effect = fake_loader

    from neutral_coupling.genex_coupling.genex_interface import get_tau_for_vars

    result = get_tau_for_vars("path", ["e", "D"])

    assert result is None


@patch("neutral_coupling.genex_coupling.genex_interface.load_snaps_genex")
def test_get_tau_for_vars_exception(mock_load):
    mock_load.side_effect = RuntimeError("mid-write")

    from neutral_coupling.genex_coupling.genex_interface import get_tau_for_vars

    result = get_tau_for_vars("path", ["e"])

    assert result is None

@patch("neutral_coupling.genex_coupling.genex_interface.get_tau_for_vars")
@patch("neutral_coupling.genex_coupling.genex_interface.get_mom_paths")
def test_wait_until_genex_stable_success(mock_paths, mock_tau):
    tau = np.array([0.0, 1.0])

    mock_paths.return_value = ["dummy"]

    # simulate: None → tau → tau (stable)
    mock_tau.side_effect = [
        None,
        tau,
        tau,
        tau,
    ]

    from neutral_coupling.genex_coupling.genex_interface import wait_until_genex_stable

    result = wait_until_genex_stable(
        "path",
        ["e"],
        check_interval=0.0,
        stable_time=0.0,   # key: eliminate timing dependency
        timeout=1.0,
    )

    assert np.array_equal(result, tau)

@patch("neutral_coupling.genex_coupling.genex_interface.get_tau_for_vars")
@patch("neutral_coupling.genex_coupling.genex_interface.get_mom_paths")
def test_wait_until_genex_stable_timeout(mock_paths, mock_tau):
    mock_paths.return_value = ["dummy"]
    mock_tau.return_value = None

    from neutral_coupling.genex_coupling.genex_interface import wait_until_genex_stable

    with pytest.raises(RuntimeError, match="did not stabilize"):
        wait_until_genex_stable(
            "path",
            ["e"],
            check_interval=0.0,
            stable_time=0.0,
            timeout=0.01,
        )
