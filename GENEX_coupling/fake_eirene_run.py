import numpy as np
from pathlib import Path
from write_netcdf import write_sources_nc
from genex_eirene_coupling import main
from temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
)
from CheckedRunEirene import CheckedRunEirene
from CheckedRunEirene import (
    COLLISION_TEMPERATURE_CONFIG,
    DEFAULT_TEMPERATURE_COLLISIONS,
    checked_collision_mappers,
    checked_sparse_temperature_handler,
    validate_temperature_collisions,
)
import os, signal, psutil
from functools import partial
from netCDF4 import Dataset
import time
import subprocess
import cProfile
import pstats
import dask
import traceback
import sys
import pdb

dask.config.set(scheduler='synchronous')

def test_full_coupling(
    tmp_path,
    override,
    fake=-1,
    temperature_mode="summed",
    collision_types=None,
    genex_path = "/pscratch/sd/j/jonroelt/source_testing/3D_source/full_workflow_test",
):
    from types import SimpleNamespace
    print("=== PYTHON ENTRY REACHED ===", flush=True)
    pid = int(Path("genex.pid").read_text())
    eirene_path = genex_path + "/eirene_setup_files"
    if fake==-1:
        genex_path = "/pscratch/sd/j/jonroelt/D3D_184833_t4800ms_try10/"
    elif fake==0:
        genex_path = "/pscratch/sd/j/jonroelt/source_testing/3D_source/test_script_wo_genex_my_data"

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
            filepattern=tmp_path + "input_sources_",
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

        #if fake:
        profiler = cProfile.Profile()
        profiler.enable()
        print("Calling main...", flush=True)
        main(args, deps=deps)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime")
        stats.print_stats(30)
        # else:
        #     main(args, deps=deps)

    except Exception:
        print("Python error: canceling job...", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        cancel_slurm_job()
        kill_tree(pid)
        raise  # preserve failure for Slurm

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

    if mode != "summed":
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


# class DensityTracker:
#     def __init__(self):
#         self.prev_norm = None
#         self.history = []

#     def check(self, n):
#         import numpy as np

#         assert np.isfinite(n).all(), "NaNs in GENE-X density"

#         norm = np.linalg.norm(n)
#         self.history.append(norm)

#         if self.prev_norm is not None:
#             assert norm != self.prev_norm, "Density not evolving"

#         self.prev_norm = norm

# def checked_run_eirene(edat, timeout, eirene_path):
#     status_code = fake_run_eirene(edat, eirene_path)

#     # --- sanity checks on EIRENE outputs ---
#     fort100 = eirene_path / "fort.100"
#     fort105 = eirene_path / "fort.105"

#     for f in (fort100, fort105):
#         with open(f) as fh:
#             for line in fh:
#                 if "E+" in line or "E-" in line:
#                     val = float(line.split()[-1])
#                     assert np.isfinite(val), f"NaN/inf in {f}"
#     tracker.check()
#     return status_code



# def fake_run_eirene(edat, eirene_path):
#     # 1. Extract interpolated density from fort.31 or cached state
#     edat_local = eirene()
#     edat_local.read_ft31(eirene_path / Path("fort.31"))
#     n = edat_local.fort31["na"]
#     R = edat.triangle_mesh.incenter[0,:]
#     Z = edat.triangle_mesh.incenter[1,:]

#     counter *= -1

#     # 2. Build synthetic source
#     R0, Z0 = 2.272, 0.0
#     sigmasq_z = 0.01
#     sigmasq_r = 0.00003
#     dt = 1e-7

#     gaussian = np.exp(-((R - R0)**2)/sigmasq_r - ((Z - Z0)**2)/sigmasq_z)
#     source = n * gaussian * counter / dt

#     # 3. Write fort files
#     write_fort_source(eirene_path / "fort.100", source, "ELECTRONS")

#     write_fort_source(eirene_path / "fort.105", source, "ions")

#     return status.SUCCESS
