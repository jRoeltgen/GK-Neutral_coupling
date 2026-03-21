import numpy as np
from unittest.mock import patch
import pytest
from hypothesis import given, strategies as st
from interpolate import (
    interp_moments,
    interpolate_source,
    interpolate_all_sources
)

from unittest.mock import patch

def test_interpolate_all_sources_dispatch_and_structure():
    tria = type("MockTria", (), {})()

    source_dict = {
        "mom1": {
            "speciesA": {
                "src1": np.array([1.0, 2.0]),
                "src2": np.array([3.0, 4.0]),
            },
            "speciesB": {
                "src3": np.array([5.0, 6.0]),
            },
        }
    }

    grid_r = np.array([0.1])
    grid_z = np.array([0.2])

    with patch("interpolate.interpolate_source", side_effect=lambda *args,
               **kwargs: "X") as mock_interp:
        out = interpolate_all_sources(
            tria,
            source_dict,
            grid_r,
            grid_z,
            method="linear",
            fill_mode="constant",
            fill_value=0.0,
        )

    # Structure preserved
    assert "mom1" in out
    assert "speciesA" in out["mom1"]
    assert "speciesB" in out["mom1"]

    # Correct number of calls (3 leaf sources)
    assert mock_interp.call_count == 3

    # All leaves mapped
    assert out["mom1"]["speciesA"]["src1"] == "X"
    assert out["mom1"]["speciesA"]["src2"] == "X"
    assert out["mom1"]["speciesB"]["src3"] == "X"

def test_interpolate_source_exact_recovery():
    # Simple triangle mesh (3 points)
    tria = type("MockTria", (), {})()
    tria.incenter = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    source = np.array([1.0, 2.0, 3.0])

    # Interpolate exactly at the same points
    grid_r = tria.incenter[:, 0]
    grid_z = tria.incenter[:, 1]

    result = interpolate_source(tria, source, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0)

    assert np.allclose(result, source)

def test_interpolate_source_linear_field():
    tria = type("MockTria", (), {})()
    tria.incenter = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    # Define linear function f(r,z) = 2r + 3z
    source = 2 * tria.incenter[:, 0] + 3 * tria.incenter[:, 1]

    grid_r = np.array([0.25, 0.5])
    grid_z = np.array([0.25, 0.5])

    result = interpolate_source(tria, source, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0)

    expected = 2 * grid_r + 3 * grid_z
    assert np.allclose(result, expected)

def test_interpolate_source_outside_domain_zero():
    tria = type("MockTria", (), {})()
    tria.incenter = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    source = np.array([1.0, 2.0, 3.0])

    # Clearly outside convex hull
    grid_r = np.array([10.0])
    grid_z = np.array([10.0])

    result = interpolate_source(tria, source, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0)

    assert result[0] == 0.0

def test_interpolate_source_shape():
    tria = type("MockTria", (), {})()
    tria.incenter = np.random.rand(10, 2)
    source = np.random.rand(10)

    grid_r = np.random.rand(6)
    grid_z = np.random.rand(6)

    result = interpolate_source(tria, source, grid_r, grid_z, method="linear",
                            fill_mode="constant", fill_value=0.0)

    assert result.shape == grid_r.shape

@pytest.fixture
def interp_inputs():
    gmtry = {
        "crx": np.array([[[0, 10, 20, 30]]]),   # shape (1,1,4)
        "cry": np.array([[[0, 100, 200, 300]]]),
    }

    grid_r = np.array([1.0])
    grid_z = np.array([2.0])

    field = np.array([5.0])

    return gmtry, grid_r, grid_z, field


@pytest.mark.parametrize(
    "ind, expected_r, expected_z",
    [
        ([0, 2], 10.0, 100.0),
        ([2, 3], 25.0, 250.0),
        ([0, 1, 2, 3], 15.0, 150.0),
    ],
)
@patch("interpolate.griddata")
def test_interp_moments_indices(mock_griddata, interp_inputs,
                               ind, expected_r, expected_z):

    gmtry, grid_r, grid_z, field = interp_inputs
    mock_griddata.return_value = np.zeros((1, 1))

    interp_moments(gmtry, grid_r, grid_z, field, ind)

    args, _ = mock_griddata.call_args
    r_passed, z_passed = args[2]

    assert np.allclose(r_passed, [[expected_r]])
    assert np.allclose(z_passed, [[expected_z]])

def test_interp_moments_linear_field():
    # Geometry (simple but nontrivial)
    gmtry = {
        "crx": np.array([[[0, 1, 0, 1]]]),   # (1,1,4)
        "cry": np.array([[[0, 0, 1, 1]]]),
    }

    # Define interpolation points
    grid_r = np.array([0, 1, 0, 1])
    grid_z = np.array([0, 0, 1, 1])

    # Linear field: f = r + 2z
    field = grid_r + 2 * grid_z

    result = interp_moments(gmtry, grid_r, grid_z, field, ind=[0, 1, 2, 3])

    # Expected interpolation point
    r_expected = np.mean(gmtry["crx"][0, 0, :])
    z_expected = np.mean(gmtry["cry"][0, 0, :])

    expected = r_expected + 2 * z_expected

    assert np.allclose(result[0, 0], expected)

@given(
    nx=st.integers(min_value=1, max_value=5),
    ny=st.integers(min_value=1, max_value=5),
    npts=st.integers(min_value=4, max_value=20),
)
def test_interp_moments_properties(nx, ny, npts):

    # Geometry
    crx = np.random.rand(nx, ny, 4)
    cry = np.random.rand(nx, ny, 4)

    gmtry = {"crx": crx, "cry": cry}

    # Interpolation points
    grid_r = np.random.rand(npts)
    grid_z = np.random.rand(npts)

    field = np.random.rand(npts)

    result = interp_moments(gmtry, grid_r, grid_z, field, ind=[0, 1, 2, 3])

    # ---- Properties ----
    assert result.shape == (nx, ny)
    assert not np.isnan(result).any()
    assert np.isfinite(result).all()

@patch("interpolate.griddata")
def test_interp_moments_warns_on_nonstandard_ind(mock_griddata):
    gmtry = {
        "crx": np.random.rand(1, 1, 4),
        "cry": np.random.rand(1, 1, 4),
    }

    grid_r = np.array([0.0])
    grid_z = np.array([0.0])
    field = np.array([1.0])

    mock_griddata.return_value = np.zeros((1, 1))

    with pytest.warns(UserWarning, match="non-standard ind"):
        interp_moments(gmtry, grid_r, grid_z, field, ind=[1, 3])