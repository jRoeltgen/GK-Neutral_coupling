import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
from torx.normalization.normalization_m import Normalization
from pint import UnitRegistry
ureg = UnitRegistry()

from collections import defaultdict

from genex_interface import (
    initialize_genex,
    load_latest_genex_fields,
    calculate_temperatures,
    toroidal_avg,
)

@pytest.fixture
def fake_norm():
    return Normalization({
        "Te0": 0.04 * ureg.keV,
        "Ti0": 0.04 * ureg.keV,
        "n0": 1.0e19 / (ureg.meter**(3)),
        "c_s0": 1.0 * ureg.meter / ureg.second,
        "elementary_charge": 1.0 * ureg.e,
        "Mi": 2.0 * ureg.u
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

@patch("genex_interface.initialize_genex_from_filepath")
def test_initialize_genex(mock_init):
    mock_init.return_value = ("grid", "equi", "params", "norm")

    result = initialize_genex("path")

    assert result == ("grid", "equi", "params", "norm")
    mock_init.assert_called_once_with("path")

@patch("genex_interface.parallel_temperature")
@patch("genex_interface.perpendicular_temperature")
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

    assert np.allclose(Ttot, Tpar + Tperp)
    assert Ttot.attrs["norm"] == 123

    mock_par.assert_called_once()
    mock_perp.assert_called_once()

@patch("genex_interface.calculate_temperatures")
@patch("genex_interface.electrostatic_ExB_heat_flux")
@patch("genex_interface.velocities_m")
@patch("genex_interface.electric_field")
@patch("genex_interface.load_snaps_genex")
def test_load_latest_genex_fields(
    mock_load,
    mock_efield,
    mock_vel,
    mock_q,
    mock_calc_temp,
    fake_grid,
    fake_norm,
    fake_data,
):

    # ---- mock data loading ----
    def fake_loader(path, spec, field):
        da = fake_data.copy()
        da = da.expand_dims(tau=[0, 1])  # simulate time dimension
        return da

    mock_load.side_effect = fake_loader

    # ---- mock derived functions ----
    mock_efield.return_value = "efield"
    mock_vel.ExB_velocity.return_value = 1.0
    mock_vel.diamagnetic_velocity.return_value = 2.0
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

    spec = ["D", "C"]

    out = load_latest_genex_fields(
        gpath="path",
        spec=spec,
        grid=fake_grid,
        equi="equi",
        params="params",
        norm=fake_norm,
    )

    # ---- structure checks ----
    assert "n" in out
    assert "u_par" in out
    assert "Ttot" in out
    assert "u_rad" in out

    for s in spec:
        assert s in out["n"]
        assert s in out["Ttot"]
        assert s in out["u_rad"]

    # es_pot special case
    assert "N/A" in out["es_pot"]

    # radial velocity combination
    for s in spec:
        assert out["u_rad"][s] == 1.0 + 2.0

    # ensure temperature calculation called
    assert mock_calc_temp.call_count == len(spec)

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