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
    species,
    get_genex_species,
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

@pytest.fixture
def fake_species():
    return [
        species("e", -1),   # electron
        species("D", +1),   # ion
    ]

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

def test_load_latest_genex_fields(
    fake_grid,
    fake_norm,
    fake_data,
    fake_species,
):
    with (
        patch("genex_interface.load_snaps_genex") as mock_load,
        patch("genex_interface.electric_field") as mock_efield,
        patch("genex_interface.velocities_m") as mock_vel,
        patch("genex_interface.electrostatic_ExB_heat_flux") as mock_q,
        patch("genex_interface.calculate_temperatures") as mock_calc_temp,
        patch("genex_interface.total_pressure") as mock_total_pressure
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

        out = load_latest_genex_fields(
            gpath="path",
            all_spec=fake_species,
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
        assert "pr" in out

        spec = []
        for sp in fake_species:
            spec.append(sp.name)
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
        )

def test_species_classification():
    e = species("e", -1)
    d = species("D+", 1)
    n = species("D0", 0)

    assert e.name == "e"
    assert e.charge == -1
    assert e.is_electron is True

    assert d.name == "D+"
    assert d.charge == 1
    assert d.is_electron is False

    assert n.is_electron is False

@patch("genex_interface.f90nml.read")
def test_get_genex_species_basic(mock_read, tmp_path):
    mock_read.return_value = {
        "params_species": {
            "names": ["e", "D"],
            "charge": [-1, 1],
        }
    }

    result = get_genex_species(tmp_path)

    assert len(result) == 2
    assert result[0].name == "e"
    assert result[0].is_electron is True
    assert result[1].name == "D"
    assert result[1].is_electron is False

@patch("genex_interface.f90nml.read")
def test_get_genex_species_ignores_empty(mock_read, tmp_path):
    mock_read.return_value = {
        "params_species": {
            "names": ["e", " ", "", "D"],
            "charge": [-1, 0, 0, 1],
        }
    }

    result = get_genex_species(tmp_path)

    assert len(result) == 2
    names = [s.name for s in result]
    assert names == ["e", "D"]

@patch("genex_interface.f90nml.read")
def test_get_genex_species_charge_alignment(mock_read, tmp_path):
    mock_read.return_value = {
        "params_species": {
            "names": ["e", "D"],
            "charge": [-2, 3],  # unusual but valid
        }
    }

    result = get_genex_species(tmp_path)

    assert result[0].charge == -2
    assert result[1].charge == 3