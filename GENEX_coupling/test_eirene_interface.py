import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import subprocess

from eirene_interface import (
    eirene_interface,
    get_temperatures,
    map_ion_temperatures_to_triangles,
    prepare_fort31,
    dict_to_array,
    run_eirene,
    status,
)

@pytest.fixture
def fake_eirene_path(tmp_path):
    # Create a temporary directory to simulate the eirene path
    return tmp_path

@pytest.fixture
def fake_b2_path(tmp_path):
    # B2 path can be empty; B2IO is mocked
    return tmp_path

@patch("eirene_interface.EireneInputParser")
@patch("eirene_interface.triangle_mesh.triangle_mesh")
@patch("eirene_interface.B2IO.B2")
@patch("eirene_interface.eireneIO.eirene")
def test_eirene_interface(mock_eirene_class, mock_b2_class,
                          mock_mesh_class, mock_parser_class,
                          tmp_path):

    eirene_path = tmp_path
    b2_path = tmp_path

    # ---- Mocks ----
    fake_eirene = MagicMock()
    fake_b2 = MagicMock()
    fake_mesh = MagicMock()
    fake_parser = MagicMock()

    mock_eirene_class.return_value = fake_eirene
    mock_b2_class.return_value = fake_b2
    mock_mesh_class.return_value = fake_mesh
    mock_parser_class.return_value = fake_parser

    # Geometry dimensions now come from edat
    fake_eirene.plasma_gmtry = {"nx": 10, "ny": 15}
    fake_eirene.fort31 = {
        "fnax": np.array([[0.0, 1.0], [0.0, 2.0]]),
        "fnay": np.array([[1.0, 0.0], [0.0, 2.0]]),
    }

    # Species parsing determines ns
    fake_parser.species = {"bulk_ions": ["D", "T", "He"]}  # ns = 3

    # ---- Call ----
    edat, b2dat, pol_mask, rad_mask = eirene_interface(eirene_path, b2_path)

    # ---- Assertions ----
    assert edat is fake_eirene
    assert b2dat is fake_b2

    # Parser usage
    mock_parser_class.assert_called_once_with(eirene_path / "input.dat")
    fake_parser.parse_species.assert_called_once()

    # Mesh setup
    mock_mesh_class.assert_called_once_with(eirene_path)
    fake_mesh.calc_incenter.assert_called_once()

    # File reads
    fake_eirene.read_ft30.assert_called_once_with(eirene_path / "fort.30")
    fake_eirene.read_ft31.assert_called_once_with(
        eirene_path / "fort.31", fake_eirene.plasma_gmtry["nx"]+2,
        fake_eirene.plasma_gmtry["ny"]+2, len(fake_parser.species["bulk_ions"])
    )

    # B2 init
    mock_b2_class.assert_called_once_with(b2_path)

    np.testing.assert_array_equal(
        pol_mask, np.array([[True, False], [True, False]])
    )
    np.testing.assert_array_equal(
        rad_mask, np.array([[False, True], [True, False]])
    )


def test_map_ion_temperatures_directly_to_triangles():
    from scipy.spatial import Delaunay

    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    edat = MagicMock()
    edat.triangle_mesh.incenter = np.array([
        [0.25, 0.25],
        [2.0, 2.0],
    ])

    result = map_ion_temperatures_to_triangles(
        edat,
        {"D+": points[:, 0] + points[:, 1]},
        Delaunay(points),
        scale=1.0,
    )

    np.testing.assert_allclose(result["Ti_D+"], np.array([0.5, 0.0]))


@pytest.fixture
def complete_temperature_edat():
    edat = MagicMock()
    edat.triangle_mesh.incenter = np.array([[0.0, 0.0], [1.0, 0.0]])
    edat.masses = {
        "atoms": [2.0],
        "molecules": [4.0],
        "test_ions": [6.0],
    }
    edat.fort46 = {
        "atom labels": ["D\n"],
        "molecule labels": ["D2\n"],
        "ion labels": ["D2+\n"],
        "pdena": np.array([[1.0], [2.0]]),
        "pdenm": np.array([[1.0], [2.0]]),
        "pdeni": np.array([[1.0], [2.0]]),
        "edena": np.array([[10.0], [20.0]]),
        "edenm": np.array([[30.0], [40.0]]),
        "edeni": np.array([[50.0], [60.0]]),
        "vxdena": np.array([[2.0], [3.0]]),
        "vxdenm": np.array([[2.0], [3.0]]),
        "vxdeni": np.array([[2.0], [3.0]]),
        "vydena": np.zeros((2, 1)),
        "vzdena": np.zeros((2, 1)),
        "vydenm": np.zeros((2, 1)),
        "vzdenm": np.zeros((2, 1)),
        "vydeni": np.zeros((2, 1)),
        "vzdeni": np.zeros((2, 1)),
    }
    return edat


def test_get_temperatures_applies_handlers_to_every_species_class(
    tmp_path, complete_temperature_edat
):
    sparse_calls = []
    pseudo_calls = []

    def sparse_handler(temperature, density, points, *, threshold):
        sparse_calls.append(
            (temperature.copy(), density.copy(), points.copy(), threshold)
        )
        return temperature + 10.0

    def pseudo_handler(
        raw_temperatures,
        density_block,
        *,
        particle_class,
        species_labels,
    ):
        pseudo_calls.append(
            (
                [value.copy() for value in raw_temperatures],
                density_block.copy(),
                particle_class,
                species_labels,
            )
        )
        return raw_temperatures[0] + 100.0, density_block[:, 0]

    result = get_temperatures(
        complete_temperature_edat,
        tmp_path,
        sparse_temperature_handler=sparse_handler,
        pseudo_temperature_handler=pseudo_handler,
        sparse_threshold=0.25,
    )

    physical_labels = {"Tn_D", "Tm_D2", "Tti_D2+"}
    pseudo_labels = {"Tn_ATOMS", "Tm_MOLECULES", "Tti_TEST IONS"}
    assert set(result) == physical_labels | pseudo_labels
    assert len(sparse_calls) == 6
    assert len(pseudo_calls) == 3
    assert {call[2] for call in pseudo_calls} == {
        "atoms", "molecules", "test_ions"
    }
    assert all(call[3] == 0.25 for call in sparse_calls)
    assert all(
        np.array_equal(call[2], complete_temperature_edat.triangle_mesh.incenter)
        for call in sparse_calls
    )

    expected_physical = {
        "Tn_D": np.array([22.0, 27.5]),
        "Tm_D2": np.array([46.0, 46.5]),
        "Tti_D2+": np.array([70.0, 65.5]),
    }
    for label, expected in expected_physical.items():
        np.testing.assert_allclose(result[label], expected)

    for physical, pseudo in zip(
        ("Tn_D", "Tm_D2", "Tti_D2+"),
        ("Tn_ATOMS", "Tm_MOLECULES", "Tti_TEST IONS"),
    ):
        np.testing.assert_allclose(result[pseudo], result[physical] + 100.0)


@pytest.mark.parametrize("include_ions", [False, True])
def test_get_temperatures_adds_ions_only_when_values_are_given(
    tmp_path, complete_temperature_edat, monkeypatch, include_ions
):
    mapped = {"Ti_D+": np.array([70.0, 80.0])}
    mapper = MagicMock(return_value=mapped)
    monkeypatch.setattr(
        "eirene_interface.map_ion_temperatures_to_triangles", mapper
    )
    ion_values = {"D+": np.array([7.0, 8.0])} if include_ions else None
    triangulation = object() if include_ions else None

    result = get_temperatures(
        complete_temperature_edat,
        tmp_path,
        ion_temperature_values=ion_values,
        genex_triangulation=triangulation,
    )

    if include_ions:
        np.testing.assert_array_equal(result["Ti_D+"], mapped["Ti_D+"])
        mapper.assert_called_once_with(
            complete_temperature_edat, ion_values, triangulation
        )
    else:
        assert not any(label.startswith("Ti_") for label in result)
        mapper.assert_not_called()


def test_get_temperatures_requires_triangulation_with_ion_values(
    tmp_path, complete_temperature_edat
):
    with pytest.raises(ValueError, match="genex_triangulation"):
        get_temperatures(
            complete_temperature_edat,
            tmp_path,
            ion_temperature_values={"D+": np.array([7.0, 8.0])},
        )

def test_prepare_fort31_3D():
    edat = MagicMock()

    nx, ny, ni = 4, 5, 2

    # Only include fields actually used by prepare_fort31
    edat.fort31 = {
        "ua": np.zeros((nx, ny, ni)),
        "bb": np.ones((nx, ny, 1)),
        "na": np.zeros((nx, ny, ni)),
        "ww": np.zeros((nx, ny, ni)),
        "te": np.zeros((nx, ny)),
        "ti": np.zeros((nx, ny, ni)),
        "fnax": np.zeros((nx, ny, ni)),
        "fnay": np.zeros((nx, ny, ni)),
        "uadia": np.zeros((nx, ny)),
        "vadia": np.zeros((nx, ny)),
        "po": np.zeros((nx, ny)),
        "pr": np.zeros((nx, ny)),
        "fhex": np.zeros((nx, ny)),
        "fhix": np.zeros((nx, ny)),
        "vv": np.zeros((nx, ny, ni)),   # needed
        "up": np.zeros((nx, ny, ni)),   # needed
    }

    # CRITICAL: prevent full write validation
    edat.write_ft31 = MagicMock()

    genex_data = {
        "n": {"D": np.ones((nx, ny)), "T": 2*np.ones((nx, ny))},
        "u_par": {"D": np.ones((nx, ny)), "T": np.ones((nx, ny))},
        "u_rad": {"D": np.ones((nx, ny)), "T": np.ones((nx, ny))},
        "u_phi": {"D": np.ones((nx, ny)), "T": np.ones((nx, ny))},
        "Ttot": {"electrons": np.ones((nx,ny)), "D": np.ones((nx, ny)),
                 "T": np.ones((nx, ny))},
        "es_pot": {"arb.": np.ones((nx, ny))},
        "pr": {"arb.": np.ones((nx, ny))},
        "fnax": {"D": np.ones((nx, ny)), "T": 2*np.ones((nx, ny))},
        "fnay": {"D": np.ones((nx, ny)), "T": np.ones((nx, ny))},
        "Q_par": {"electrons": np.ones((nx,ny)), "D": np.ones((nx, ny)),
                  "T": np.ones((nx, ny))}
    }

    prepare_fort31(edat, genex_data, ["electrons"], ["D","T"])

    # ---- Assertions ----

    # Density mapping
    assert np.all(edat.fort31["na"][:,:,0] == 1)
    assert np.all(edat.fort31["na"][:,:,1] == 2)

    # upar stored
    assert np.all(edat.fort31["ua"] == 1)

    # upol = upar * bb[:,:,0] (bb=1)
    assert np.all(edat.fort31["up"] == edat.fort31["ua"])

    # radial velocity propagated
    assert np.all(edat.fort31["vv"] == 1)

    # fnax = upol * na
    assert np.all(edat.fort31["fnax"] == edat.fort31["na"])

    # Heat flux shape sanity
    assert edat.fort31["fhix"].shape == (nx, ny)

def test_prepare_fort31_2D():
    edat = MagicMock()

    nx, ny, ni = 4, 5, 1

    # Only include fields actually used by prepare_fort31
    edat.fort31 = {
        "ua": np.zeros((nx, ny)),
        "bb": np.ones((nx, ny, 1)),
        "na": np.zeros((nx, ny)),
        "ww": np.zeros((nx, ny)),
        "te": np.zeros((nx, ny)),
        "ti": np.zeros((nx, ny)),
        "fnax": np.zeros((nx, ny)),
        "fnay": np.zeros((nx, ny)),
        "uadia": np.zeros((nx, ny)),
        "vadia": np.zeros((nx, ny)),
        "po": np.zeros((nx, ny)),
        "pr": np.zeros((nx, ny)),
        "fhex": np.zeros((nx, ny)),
        "fhix": np.zeros((nx, ny)),
        "vv": np.zeros((nx, ny)),
        "up": np.zeros((nx, ny)),
    }

    edat.write_ft31 = MagicMock()

    genex_data = {
        "n": {"D": np.ones((nx, ny))},
        "u_par": {"D": np.ones((nx, ny))},
        "u_rad": {"D": np.ones((nx, ny))},
        "u_phi": {"D": np.ones((nx, ny))},
        "Ttot": {"electrons": np.ones((nx,ny)), "D": np.ones((nx, ny))},
        "es_pot": {"arb.": np.ones((nx, ny))},
        "pr": {"arb.": np.ones((nx, ny))},
        "fnax": {"D": np.ones((nx, ny))},
        "fnay": {"D": np.ones((nx, ny))},
        "Q_par": {"electrons": np.ones((nx,ny)), "D": np.ones((nx, ny))}
    }

    prepare_fort31(edat, genex_data, ["electrons"], ["D"])

    # ---- Assertions ----

    # Density mapping
    assert np.all(edat.fort31["na"][:,:] == 1)

    # upar stored
    assert np.all(edat.fort31["ua"] == 1)

    # upol = upar * bb[:,:,0] (bb=1)
    assert np.all(edat.fort31["up"] == edat.fort31["ua"])

    # radial velocity propagated
    assert np.all(edat.fort31["vv"] == 1)

    # fnax = upol * na
    assert np.all(edat.fort31["fnax"] == edat.fort31["na"])

    # Heat flux shape sanity
    assert edat.fort31["fhix"].shape == (nx, ny)

def test_dict_to_array_ordering():
    data = {
        "A": np.ones((2,2)),
        "B": 2*np.ones((2,2))
    }

    arr = dict_to_array(data, ["A","B"])

    assert arr.shape == (2,2,2)
    assert np.all(arr[:,:,0] == 1)
    assert np.all(arr[:,:,1] == 2)

def test_dict_to_array_dimension_mismatch():
    data = {
        "A": np.ones((2,2)),
        "B": np.ones((3,3))
    }

    with pytest.raises(ValueError):
        dict_to_array(data, ["A","B"])

@patch("eirene_interface.subprocess.Popen")
def test_run_eirene_success(mock_popen, tmp_path):
    mock_proc = mock_popen.return_value
    mock_proc.wait.return_value = None
    mock_proc.returncode = 0

    result = run_eirene(10, tmp_path, command="test_cmd")

    assert result == status.SUCCESS
    mock_popen.assert_called_once()

@patch("eirene_interface.subprocess.Popen")
def test_run_eirene_failure(mock_popen, tmp_path):
    mock_proc = mock_popen.return_value
    mock_proc.wait.return_value = None
    mock_proc.returncode = 1

    result = run_eirene(10, tmp_path)

    assert result == status.ERROR

@patch("eirene_interface.os.killpg")
@patch("eirene_interface.subprocess.Popen")
def test_run_eirene_timeout(mock_popen, mock_killpg, tmp_path):
    mock_proc = mock_popen.return_value
    mock_proc.pid = 1234  # critical

    mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=10)

    result = run_eirene(10, tmp_path)

    assert result == status.TIMEOUT
    mock_killpg.assert_called()  # optional but good

@patch("eirene_interface.os.killpg")
@patch("eirene_interface.subprocess.Popen")
def test_run_eirene_timeout_kills_process(mock_popen, mock_killpg, tmp_path):
    mock_proc = mock_popen.return_value
    mock_proc.pid = 1234

    # First wait → timeout
    # Second wait → also timeout (forces SIGKILL path)
    mock_proc.wait.side_effect = [
        subprocess.TimeoutExpired(cmd="x", timeout=10),
        subprocess.TimeoutExpired(cmd="x", timeout=10),
    ]

    result = run_eirene(10, tmp_path)

    assert result == status.TIMEOUT
    assert mock_killpg.call_count >= 1

@patch("eirene_interface.subprocess.Popen",
       side_effect=FileNotFoundError)
def test_run_eirene_not_found(mock_popen, tmp_path):

    result = run_eirene(10, tmp_path)

    assert result == status.ERROR
