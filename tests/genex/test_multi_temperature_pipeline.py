from types import SimpleNamespace
from pathlib import Path
import sys

from netCDF4 import Dataset
import numpy as np
import pytest

INTEGRATION_DIR = Path(__file__).resolve().parent / "integration"
sys.path.insert(0, str(INTEGRATION_DIR))

from imitation_eirene_checker import (
    COLLISION_TEMPERATURE_CONFIG,
    DEFAULT_TEMPERATURE_COLLISIONS,
    checked_collision_mappers,
    checked_sparse_temperature_handler,
    requested_strata_output_order,
    validate_temperature_collisions,
    CheckedRunEirene,
)
from neutral_coupling.common.eirene_io import eirene
from neutral_coupling.common.source_post_processor import SourcePostProcessor
from neutral_coupling.genex_coupling.genex_eirene_coupling import (
    interpolate_all_sources_wrapper,
    interpolate_temperature_values,
)
from imitation_eirene_driver import checked_write_nc


@pytest.mark.parametrize(
    "collision_types",
    [
        DEFAULT_TEMPERATURE_COLLISIONS,
        ("atom-plasma", "plasma-plasma"),
        ("molecule-plasma", "testion-plasma"),
    ],
)
def test_multiple_temperature_pipeline_writes_exact_spatial_fields(
    tmp_path, collision_types
):
    x, z = np.meshgrid(
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.5, 1.0]),
    )
    points = np.column_stack([x.ravel(), z.ravel()])
    triangle_mesh = SimpleNamespace(incenter=points)
    compute = np.ones(points.shape[0], dtype=bool)

    all_collisions = DEFAULT_TEMPERATURE_COLLISIONS
    centers = {
        collision: points[index]
        for index, collision in enumerate(all_collisions)
    }
    sources = {"particle": {}}
    temperatures = {}
    expected_peak = {}

    for index, collision in enumerate(collision_types):
        center = centers[collision]
        distance_squared = np.sum((points - center) ** 2, axis=1)
        source = (index + 1.0) * np.exp(-distance_squared / 0.02)
        sources["particle"][collision] = {"D": {"SUM": source}}
        expected_peak[collision] = int(np.argmax(source))

        prefix = COLLISION_TEMPERATURE_CONFIG[collision]["prefix"]
        label = f"{prefix}D"
        temperatures[label] = (
            10.0 * (index + 1)
            + (index + 1) * points[:, 0]
            + (index + 2) * points[:, 1]
        )

    processor = SourcePostProcessor(
        sources,
        temperatures,
        collision_mappers=checked_collision_mappers(collision_types),
    )
    grouped_sources, used_temperatures = processor.regroup_by_temperature()

    interpolated_sources = interpolate_all_sources_wrapper(
        triangle_mesh,
        grouped_sources,
        points[:, 0],
        points[:, 1],
        compute,
    )
    direct_values = {
        label: values
        for label, values in temperatures.items()
        if label.startswith("Ti_")
    }
    interpolated_temperatures = interpolate_temperature_values(
        triangle_mesh,
        used_temperatures,
        points[:, 0],
        points[:, 1],
        compute,
        direct_values=direct_values,
    )

    output = tmp_path / "multiple_temperatures.nc"
    checked_write_nc(
        output,
        interpolated_sources,
        dim_RZ=points.shape[0],
        temperature_values=interpolated_temperatures,
        write_temperature=True,
        mode="multiple_temperatures",
        collision_types=collision_types,
    )

    expected_labels = {
        f"{COLLISION_TEMPERATURE_CONFIG[collision]['prefix']}D"
        for collision in collision_types
    }
    assert set(interpolated_temperatures) == expected_labels
    assert {
        label
        for species_sources in interpolated_sources["particle"].values()
        for label in species_sources
    } == expected_labels

    observed_peaks = []
    with Dataset(output) as nc:
        ids = []
        for collision in collision_types:
            label = f"{COLLISION_TEMPERATURE_CONFIG[collision]['prefix']}D"
            group = nc.groups[f"temperature_{label}"]
            ids.append(group.temperature_id)

            written_temperature = group.variables["temperature"][:]
            expected_temperature = temperatures[label].reshape(1, -1)
            np.testing.assert_allclose(
                written_temperature,
                expected_temperature,
                rtol=1e-12,
                atol=1e-12,
            )
            assert np.ptp(written_temperature) > 0

            written_source = group.groups["D"].variables["particle"][:]
            peak = int(np.argmax(written_source))
            observed_peaks.append(peak)
            assert peak == expected_peak[collision]
            assert np.isfinite(written_source).all()
            assert np.any(written_source != 0)

        assert len(ids) == len(set(ids))
        assert len(observed_peaks) == len(set(observed_peaks))


def test_checked_temperature_collision_selection_validation():
    assert validate_temperature_collisions(
        ("atom-plasma", "plasma-plasma")
    ) == ("atom-plasma", "plasma-plasma")

    with pytest.raises(ValueError, match="at least two"):
        validate_temperature_collisions(("atom-plasma",))
    with pytest.raises(ValueError, match="Unknown"):
        validate_temperature_collisions(("atom-plasma", "not-a-collision"))
    with pytest.raises(ValueError, match="unique"):
        validate_temperature_collisions(("atom-plasma", "atom-plasma"))


def test_requested_strata_output_order_places_sum_last():
    assert requested_strata_output_order(["SUM", 22, 21]) == [
        21, 22, "SUM"
    ]


def test_checked_fort_source_writes_one_block_per_stratum(tmp_path):
    runner = CheckedRunEirene()
    source = np.array([1.0, 2.0, 3.0])
    filepath = tmp_path / "fort.105"

    runner.write_fort_source(
        filepath,
        source,
        "ions",
        strata_values=[np.zeros_like(source), source],
    )

    contents = filepath.read_text()
    assert contents.count("PARTICLE SOURCE FROM") == 2
    assert contents.count(" 3.000000E+00") == 1


def test_checked_mappers_ignore_pseudo_temperature_labels():
    mapper = checked_collision_mappers(
        ("atom-plasma", "plasma-plasma")
    )["atom-plasma"]
    source = np.array([1.0, 2.0])

    result = mapper(
        mom="particle",
        species="D+",
        strata_dict={"SUM": source},
        temperature_values={
            "Tn_D": np.array([2.0, 3.0]),
            "Tn_ATOMS": np.array([4.0, 5.0]),
        },
        context={},
    )

    assert set(result["values"]) == {"Tn_D"}
    np.testing.assert_array_equal(result["values"]["Tn_D"], source)


def test_checked_sparse_handler_fills_any_nonempty_pattern():
    temperature = np.array([7.0, np.nan, np.nan])
    density = np.array([1.0, 0.0, 0.0])
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    result = checked_sparse_temperature_handler(
        temperature,
        density,
        points,
        threshold=0.0,
    )

    np.testing.assert_array_equal(result, np.full(3, 7.0))
    with pytest.raises(ValueError, match="No positive-density"):
        checked_sparse_temperature_handler(
            temperature,
            np.zeros(3),
            points,
            threshold=1.0,
        )


def test_checked_runner_writes_spatially_varying_fort46(tmp_path):
    runner = CheckedRunEirene(mode="multiple_temperatures")
    parser = SimpleNamespace(
        species={
            "atoms": ["D"],
            "molecules": ["D2"],
            "test_ions": ["D2+"],
        }
    )
    R = np.array([1.0, 1.5, 2.0, 2.5])
    Z = np.array([-1.0, -0.5, 0.5, 1.0])
    filepath = tmp_path / "fort.46"

    runner._write_checked_fort46(filepath, parser, R, Z)

    loaded = eirene()
    loaded.read_ft46(filepath)
    assert loaded.fort46["natm"] == 1
    assert loaded.fort46["nmol"] == 1
    assert loaded.fort46["nion"] == 1
    for suffix in ("a", "m", "i"):
        temperature = (
            loaded.fort46["eden" + suffix][:, 0]
            / loaded.fort46["pden" + suffix][:, 0]
        )
        assert np.isfinite(temperature).all()
        assert np.ptp(temperature) > 0
