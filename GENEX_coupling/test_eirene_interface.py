import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from eirene_interface import eirene_interface

@pytest.fixture
def fake_eirene_path(tmp_path):
    # Create a temporary directory to simulate the eirene path
    # Optionally touch a fake fort.44 to test the branch where it exists
    (tmp_path / "fort.44").touch()
    return tmp_path

@pytest.fixture
def fake_b2_path(tmp_path):
    # B2 path can be empty; B2IO is mocked
    return tmp_path

@patch("eirene_interface.triangle_mesh.triangle_mesh")
@patch("eirene_interface.B2IO.B2")
@patch("eirene_interface.eireneIO.eirene")
def test_eirene_interface(mock_eirene_class, mock_b2_class, mock_mesh_class,
                          fake_eirene_path, fake_b2_path):
    """
    Test the orchestration in eirene_interface:
    - correct handling of fort.44 present vs absent
    - triangle_mesh calc_incenter called
    - eirene and B2 objects are returned
    """
    # ---- Setup mocks ----
    fake_eirene = MagicMock()
    fake_b2 = MagicMock()
    fake_mesh = MagicMock()

    mock_eirene_class.return_value = fake_eirene
    mock_b2_class.return_value = fake_b2
    mock_mesh_class.return_value = fake_mesh

    # Simulate fort44 meta structure
    fake_eirene.fort44 = {"meta": {"npls": 5}}
    # nx, ny extracted from B2 object
    fake_b2.gmtry = {"vol": MagicMock(shape=(10, 15))}

    # Call the interface function
    edat, b2dat = eirene_interface(fake_eirene_path, fake_b2_path)

    # ---- Assertions ----
    # Returned objects are the mocked ones
    assert edat is fake_eirene
    assert b2dat is fake_b2

    # Triangle mesh initialized with eirene path
    mock_mesh_class.assert_called_once_with(fake_eirene_path)
    assert edat.triangle_mesh is fake_mesh

    # calc_incenter called on triangle mesh
    fake_mesh.calc_incenter.assert_called_once_with()

    # B2 initialized with the b2 path
    mock_b2_class.assert_called_once_with(fake_b2_path)

    # Read fort44 called if file exists
    fake_eirene.read_ft44.assert_called_once_with(fake_eirene_path / "fort.44")

    # edat.read_ft31 called with nx, ny, ns=5 from fort44
    fake_eirene.read_ft31.assert_called_once_with(fake_eirene_path / "fort.31",
                                                  10, 15, 5)

def test_eirene_interface_no_fort44(tmp_path):
    """
    Test the branch where fort.44 does not exist.
    Should default ns=2 and still call read_ft31.
    """

    eirene_path = tmp_path
    b2_path = tmp_path

    with patch("eirene_interface.eireneIO.eirene") as mock_eirene_class, \
         patch("eirene_interface.B2IO.B2") as mock_b2_class, \
         patch("eirene_interface.triangle_mesh.triangle_mesh") as mock_mesh_class:

        fake_eirene = MagicMock()
        fake_b2 = MagicMock()
        fake_mesh = MagicMock()

        mock_eirene_class.return_value = fake_eirene
        mock_b2_class.return_value = fake_b2
        mock_mesh_class.return_value = fake_mesh

        fake_b2.gmtry = {"vol": MagicMock(shape=(3, 4))}

        # Make sure fort44 file does not exist
        edat, b2dat = eirene_interface(eirene_path, b2_path)

        # Optionally assert returned objects
        assert edat is mock_eirene_class.return_value
        assert b2dat is mock_b2_class.return_value

        # ns should default to 2
        fake_eirene.read_ft31.assert_called_once_with(eirene_path / "fort.31",
                                                      3, 4, 2)