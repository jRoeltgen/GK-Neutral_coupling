import numpy as np
from pathlib import Path
from neutral_coupling.genex_coupling.write_netcdf import write_sources_nc
from neutral_coupling.genex_coupling.genex_eirene_coupling import main
from neutral_coupling.common.temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
)
from imitation_eirene_checker import (
    CheckedRunEirene,
    COLLISION_TEMPERATURE_CONFIG,
    DEFAULT_TEMPERATURE_COLLISIONS,
    checked_collision_mappers,
    checked_sparse_temperature_handler,
    validate_temperature_collisions,
)
import os
import psutil
from functools import partial
from netCDF4 import Dataset
import time
import subprocess
import cProfile
import pstats
import dask
import traceback
import sys

dask.config.set(scheduler='synchronous')

def test_full_coupling(
    tmp_path,
    override,
    temperature_mode="summed",
    collision_types=None,
    genex_path = "/pscratch/sd/j/jonroelt/source_testing/3D_source/full_workflow_test",
    eirene_path = None,
):
    """
    Test of full coupling with analytic replacement of Eirene. Checks that files
    are passed correctly and GENE-X data is loaded correctly. If
    temperature_mode="summed", checks that the sources are updated over time.
    If temperature_mode="multiple_temperatures", checks that the multiple
    temperatures are loaded into GENE-X (by change in density).

    tmp_path - Where the sources are written to
    override - If true, overrides selection of time slice in GENE-X data. True
            selects a monotonically increasing time slice starting at 1, so
            GENE-X can be replaced with an already created set of data.
    temperature_mode - Whether to sum sources or split by temperature
    collision_types - Which Eirene collision types to use (i.e. atom-plasma)
    genex_path - Full path directory with GENE-X run data
    eirene_path - Full path to directory with eirene input files.
    """
    from types import SimpleNamespace
    print("=== PYTHON ENTRY REACHED ===", flush=True)
    pid = int(Path("genex.pid").read_text())
    if eirene_path == None:
        eirene_path = genex_path + "/eirene_setup_files"

    if temperature_mode == "multiple_temperatures":
        collision_types = validate_temperature_collisions(
            collision_types or DEFAULT_TEMPERATURE_COLLISIONS
        )
    elif temperature_mode == "summed":
        collision_types = ()
    else:
        raise ValueError(f"Unknown temperature mode: {temperature_mode}")

    checked_runner = CheckedRunEirene(
        mode=temperature_mode,
        collision_types=collision_types or None,
    )
    print(f"Using file from genex_path: {genex_path}", flush=True)
    try:
        args = SimpleNamespace(
            pid=pid,
            MAX_TIMEOUTS=1,
            eirene_time=1,
            SumTemp=temperature_mode == "summed",
            filepattern=tmp_path + "input_sources",
            genex_time_index_override=override,
            genex_path=genex_path,
            eirene_path=eirene_path,
            eirene_command="pwd",
        )

        deps = SimpleNamespace(
            run_eirene=checked_runner,
            write_nc=partial(
                checked_write_nc,
                mode=temperature_mode,
                collision_types=collision_types,
            ),
            killpg=lambda pid, sig: cancel_slurm_job(),
            sleep=lambda x: time.sleep(0.1),
            replace=os.replace,
            pid_exists=psutil.pid_exists,
            collision_mappers=(
                checked_collision_mappers(collision_types)
                if collision_types else None
            ),
            sparse_temperature_handler=(
                checked_sparse_temperature_handler
                if collision_types
                else default_sparse_temperature_handler
            ),
            pseudo_temperature_handler=(
                default_single_species_pseudo_temperature_handler
            ),
        )

        profiler = cProfile.Profile()
        profiler.enable()
        print("Calling main...", flush=True)
        main(args, deps=deps)
        profiler.disable()

        # Check if GENEX is still alive
        if psutil.pid_exists(pid):
            # This should not happen → signal a bug, not normal cleanup
            print("ERROR: main() returned but GENEX is still running", flush=True)
            cancel_slurm_job()
            kill_tree(pid)
            raise RuntimeError("Invariant violated: GENEX still running after main()")
        else:
            print("Clean completion: leaving job to exit normally.", flush=True)

        # --- post-run checks ---
        assert len(checked_runner.history) > 1

        # ensure actual variation (not just noise)
        rounded = np.round(checked_runner.history, 10)
        assert len(set(rounded)) > 1, "Density did not meaningfully change"
        if collision_types:
            for collision in collision_types:
                values = checked_runner.location_history[collision]
                assert len(values) > 1
                assert not np.allclose(
                    values[1:], values[:-1], rtol=1e-8, atol=0.0
                ), f"Density did not evolve near the {collision} source"
        else:
            checked_runner.assert_summed_source_updates_used(eirene_path)

        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime")
        stats.print_stats(30)

    except Exception:
        print("Python error: canceling job...", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        cancel_slurm_job()
        kill_tree(pid)
        raise  # preserve failure for Slurm

def checked_write_nc(
    filename,
    interp_sources,
    dim_RZ,
    temperature_values=None,
    write_temperature=False,
    *,
    mode="summed",
    collision_types=(),
):

    # --- check BEFORE writing ---
    source_by_temperature = {}
    for field, species_dict in interp_sources.items():
        for species, temperature_dict in species_dict.items():
            for temperature, values in temperature_dict.items():
                arr = np.asarray(values)
                assert np.isfinite(arr).all(), (
                    f"NaNs in interpolated sources "
                    f"({field}, {species}, {temperature})"
                )
                assert np.any(arr != 0), (
                    f"Degenerate zero field "
                    f"({field}, {species}, {temperature})"
                )
                source_by_temperature.setdefault(temperature, []).append(
                    (field, species, arr)
                )

    if mode == "summed":
        assert set(source_by_temperature) == {"SUM"}
        _check_summed_source_signs(source_by_temperature["SUM"])
    else:
        assert write_temperature
        assert temperature_values is not None
        expected_labels = {}
        for collision in collision_types:
            prefix = COLLISION_TEMPERATURE_CONFIG[collision]["prefix"]
            labels = [
                label
                for label in source_by_temperature
                if label is not None and label.startswith(prefix)
            ]
            assert len(labels) == 1, (
                f"Expected one {prefix} source group, found {labels}"
            )
            expected_labels[collision] = labels[0]

        assert set(source_by_temperature) == set(expected_labels.values())
        temperature_arrays = []
        for collision, label in expected_labels.items():
            temperature = np.asarray(temperature_values[label])
            assert temperature.shape == (1, dim_RZ)
            assert np.isfinite(temperature).all(), (
                f"{collision} temperature '{label}' contains non-finite values"
            )
            assert np.ptp(temperature) > 0, (
                f"{collision} temperature is not spatially varying"
            )
            for _, _, source in source_by_temperature[label]:
                support = np.abs(source) > 1e-5 * np.max(np.abs(source))
                assert np.any(support)
                supported_temperature = temperature[support]
                assert np.isfinite(supported_temperature).all()
                if supported_temperature.size > 1:
                    scale = np.max(np.abs(supported_temperature))
                    assert np.ptp(supported_temperature) > 1e-12 * scale, (
                        f"{collision} temperature does not vary across its "
                        "source support"
                    )
            temperature_arrays.append(temperature)

        for first in range(len(temperature_arrays)):
            for second in range(first + 1, len(temperature_arrays)):
                assert not np.allclose(
                    temperature_arrays[first],
                    temperature_arrays[second],
                    rtol=1e-8,
                    atol=0.0,
                ), "Two checked temperature groups are spatially identical"

    # call real writer
    write_sources_nc(
        filename,
        interp_sources,
        dim_RZ,
        temperature_values=temperature_values,
        write_temperature=write_temperature,
    )

    if mode == "summed":
        _check_written_summed_sources(filename, source_by_temperature["SUM"])
    else:
        with Dataset(filename) as nc:
            ids = []
            for label, entries in source_by_temperature.items():
                group = nc.groups[f"temperature_{label}"]
                ids.append(group.temperature_id)
                np.testing.assert_allclose(
                    group.variables["temperature"][:],
                    temperature_values[label],
                )
                for field, species, expected in entries:
                    np.testing.assert_allclose(
                        group.groups[species].variables[field][:],
                        expected,
                    )
            assert len(ids) == len(set(ids))


def _source_sign_summary(values):
    arr = np.asarray(values, dtype=float)
    magnitude = np.abs(arr)
    max_abs = float(np.max(magnitude))
    active = magnitude > 1e-12 * max_abs
    if not np.any(active):
        raise AssertionError("No active source cells found")
    active_values = arr[active]
    signs = np.sign(active_values)
    if np.any(signs != signs[0]):
        raise AssertionError(
            "Mixed signs found in active source cells: "
            f"min={np.min(active_values)}, max={np.max(active_values)}"
        )
    return {
        "sign": float(signs[0]),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "active_cells": int(np.count_nonzero(active)),
    }


def _check_summed_source_signs(entries):
    particle_signs = []
    for field, species, values in entries:
        if field != "particle":
            continue
        summary = _source_sign_summary(values)
        particle_signs.append(summary["sign"])
    if not particle_signs:
        raise AssertionError("No particle sources found in summed source file")
    if len(set(particle_signs)) != 1:
        raise AssertionError(
            f"Summed particle sources disagree in sign: {particle_signs}"
        )


def _check_written_summed_sources(filename, entries):
    with Dataset(filename) as nc:
        group = nc.groups["temperature_SUM"]
        for field, species, expected in entries:
            written = group.groups[species].variables[field][:]
            np.testing.assert_allclose(written, expected)

def kill_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for p in children:
            p.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass

def cancel_slurm_job():
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is not None:
        subprocess.run(["scancel", job_id])
        time.sleep(1)
    else:
        raise RuntimeError("Not running inside a SLURM job")
