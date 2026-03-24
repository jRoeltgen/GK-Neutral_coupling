import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import subprocess

from eirene_interface import (eirene_interface, prepare_fort31, dict_to_array,
                              run_eirene, status)

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

    # Species parsing determines ns
    fake_parser.species = {"bulk_ions": ["D", "T", "He"]}  # ns = 3

    # ---- Call ----
    edat, b2dat = eirene_interface(eirene_path, b2_path)

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
        eirene_path / "fort.31", 10, 15, 3
    )

    # B2 init
    mock_b2_class.assert_called_once_with(b2_path)

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