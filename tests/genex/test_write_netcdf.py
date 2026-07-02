import numpy as np
import pytest
from netCDF4 import Dataset
from hypothesis import given, strategies as st
from pathlib import Path
import tempfile
import uuid

from neutral_coupling.genex_coupling.write_netcdf import write_sources_nc


@pytest.fixture
def sample_sources():
    """
    Build a sources dictionary:
    sources[moment][species][temperature]
    """

    moments = ["density", "energy"]
    species = ["D", "C"]
    temps = ["Ti", "Te"]

    sources = {}

    for m in moments:
        sources[m] = {}
        for sp in species:
            sources[m][sp] = {}
            for t in temps:
                sources[m][sp][t] = np.ones((4, 5)) * (
                    hash((m, sp, t)) % 10
                )

    return sources


@pytest.fixture
def sample_temperatures():
    return {
        "Ti": 10.0,
        "Te": 20.0,
    }


def test_write_sources_with_temperature(tmp_path, sample_sources,
                                        sample_temperatures):

    outfile = tmp_path / "sources_with_temp.nc"

    write_sources_nc(
        outfile,
        sample_sources,
        dim_RZ=4,
        dim_phi=5,
        temperature_values=sample_temperatures,
        write_temperature=True,
    )

    with Dataset(outfile) as nc:

        # --- Check temperature groups exist
        assert "temperature_Ti" in nc.groups
        assert "temperature_Te" in nc.groups

        for i, temp in enumerate(["Ti", "Te"]):

            grp = nc.groups[f"temperature_{temp}"]

            # --- check attribute
            assert grp.temperature_id == np.int32(i+1)

            # --- check temperature variable exists
            assert "temperature" in grp.variables

            # --- check species groups
            for sp in ["D", "C"]:
                assert f"{sp}" in grp.groups

                sp_grp = grp.groups[f"{sp}"]

                for moment in ["density", "energy"]:
                    varname = f"{moment}"

                    assert varname in sp_grp.variables

                    data = sp_grp.variables[varname][:]

                    expected = sample_sources[moment][sp][temp]

                    assert np.allclose(data, expected.T)

def test_write_sources_without_temperature(tmp_path, sample_sources):

    outfile = tmp_path / "sources_no_temp.nc"

    write_sources_nc(
        outfile,
        sample_sources,
        dim_RZ=4,
        dim_phi=5,
        temperature_values=None,
        write_temperature=False,
    )

    with Dataset(outfile) as nc:

        for i, temp in enumerate(["Ti", "Te"]):

            grp = nc.groups[f"temperature_{temp}"]

            # --- attribute still exists
            assert grp.temperature_id == np.int32(i+1)

            # --- temperature variable should NOT exist
            assert "temperature" not in grp.variables

            for sp in ["D", "C"]:
                sp_grp = grp.groups[f"{sp}"]

                for moment in ["density", "energy"]:

                    varname = f"{moment}"

                    assert varname in sp_grp.variables

                    data = sp_grp.variables[varname][:]

                    expected = sample_sources[moment][sp][temp]

                    assert np.allclose(data, expected.T)

# --- Hypothesis strategy for sources dict ---
@st.composite
def sources_strategy(draw):
    identifier = st.from_regex(r"[A-Za-z][A-Za-z0-9_]{0,5}", fullmatch=True)

    moments = draw(st.lists(identifier, min_size=2, max_size=4, unique=True))
    species = draw(st.lists(identifier, min_size=2, max_size=4, unique=True))
    temps = draw(st.lists(st.one_of(identifier, st.just(None)),
                          min_size=1, max_size=3, unique=True))

    shape = draw(st.tuples(st.integers(2, 6), st.integers(2, 6)))

    sources = {}
    temperature_values = {}

    for m in moments:
        sources[m] = {}
        for sp in species:
            sources[m][sp] = {}
            for t in temps:
                arr = np.random.rand(*shape)
                sources[m][sp][t] = arr

                if t is not None:
                    temperature_values[t] = np.random.rand() * 100

    return sources, temperature_values, shape, moments, species, temps


@given(sources_strategy(), st.booleans())
def test_write_sources_property(data, write_temperature):
    sources, temperature_values, shape, moments, species, temps = data

    with tempfile.TemporaryDirectory() as tmpdir:
        outfile = Path(tmpdir) / f"test_{uuid.uuid4().hex}.nc"

        write_sources_nc(
            outfile,
            sources,
            dim_RZ=shape[0],
            dim_phi=shape[1],
            temperature_values=temperature_values if write_temperature else None,
            write_temperature=write_temperature,
        )

        with Dataset(outfile) as nc:
            # --- global dimensions
            assert "dim_RZ" in nc.dimensions
            assert "dim_phi" in nc.dimensions
            assert nc.dimensions["dim_RZ"].size == shape[0]
            assert nc.dimensions["dim_phi"].size == shape[1]

            # --- expected temperature groups (order-independent)
            expected_groups = {
                f"temperature_{t if t is not None else 'implicit'}"
                for t in temps
            }

            assert set(nc.groups.keys()) == expected_groups

            # --- build mapping: group_name -> temperature_id
            ids = {
                name: nc.groups[name].temperature_id
                for name in nc.groups
            }

            # IDs should be consecutive starting at 1
            assert sorted(ids.values()) == list(range(1, len(ids) + 1))

            # --- iterate through groups
            for temp in temps:
                grp_name = f"temperature_{temp if temp is not None else 'implicit'}"
                tgrp = nc.groups[grp_name]

                # --- temperature variable behavior
                if temp is None:
                    assert "temperature" not in tgrp.variables

                elif write_temperature:
                    assert "temperature" in tgrp.variables

                    expected_temp = np.full(
                        (shape[1], shape[0]),  # (dim_phi, dim_RZ)
                        temperature_values[temp],
                    )

                    np.testing.assert_allclose(
                        tgrp.variables["temperature"][:],
                        expected_temp,
                    )
                else:
                    assert "temperature" not in tgrp.variables

                # --- species + moments
                for sp in species:
                    spgrp = tgrp.groups[f"{sp}"]

                    for mom in moments:
                        varname = f"{mom}"
                        assert varname in spgrp.variables

                        data_nc = spgrp.variables[varname][:]
                        expected = sources[mom][sp][temp]

                        # writer stores (dim_phi, dim_RZ)
                        np.testing.assert_allclose(data_nc, expected.T)
