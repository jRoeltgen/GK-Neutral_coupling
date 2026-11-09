"""Lightweight conservative-interpolation check using a retained mesh.nc."""
from pathlib import Path
import os

import numpy as np
import pytest
import xarray as xr

from neutral_coupling.genex_coupling.interpolate import ConservativeSourceMapper


def _mesh_path():
    configured = os.environ.get("GENEX_INTERPOLATION_MESH")
    if not configured:
        pytest.skip(
            "set GENEX_INTERPOLATION_MESH to a retained GENE-X mesh.nc; "
            "the integration launchers create this artifact"
        )
    path = Path(configured)
    if not path.is_file():
        pytest.fail(f"GENEX_INTERPOLATION_MESH does not exist: {path}")
    return path


def test_conservative_interpolation_on_real_genex_mesh():
    with xr.open_dataset(_mesh_path(), group="RZ_grid") as mesh:
        r_name = "x" if "x" in mesh else "R"
        z_name = "y" if "y" in mesh else "Z"
        phi_dim = next(dim for dim in mesh[r_name].dims if "phi" in dim)
        active = (mesh["not_ghost"].isel({phi_dim: 0}).astype(bool) &
                  mesh["not_filler"].isel({phi_dim: 0}).astype(bool) &
                  (mesh["buf_zone"].isel({phi_dim: 0}) == 0))
        arrays = [np.asarray(mesh[name].isel({phi_dim: 0}))[active]
                  for name in (r_name, z_name, "jacobian", "RZw")]

    r, z, jacobian, rzw = (np.asarray(value).reshape(-1) for value in arrays)
    assert r.size >= 4
    # A bounded, spatially distributed subset keeps this test inexpensive even
    # when the retained production mesh is many gigabytes.
    selection = np.linspace(0, r.size - 1, min(r.size, 2000), dtype=int)
    points = np.column_stack((r[selection], z[selection]))
    weights = jacobian[selection] * rzw[selection]
    finite = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0)
    points, weights = points[finite], weights[finite]
    _, unique = np.unique(points, axis=0, return_index=True)
    points, weights = points[unique], weights[unique]
    assert len(points) >= 4

    mapper = ConservativeSourceMapper(points, points, weights, weights)
    fields = (
        np.ones(len(points)),
        points[:, 0] - np.median(points[:, 0]),
        np.exp(-((points[:, 0] - np.mean(points[:, 0])) ** 2 +
                 (points[:, 1] - np.mean(points[:, 1])) ** 2) /
               max(np.var(points, axis=0).sum(), np.finfo(float).eps)),
    )
    for field in fields:
        mapped, report = mapper.map(field, label="real-mesh verification")
        assert mapped.shape == field.shape
        assert np.isfinite(mapped).all()
        assert report.global_residual < 1e-12
