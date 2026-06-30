import numpy as np
from pathlib import Path
from write_netcdf import write_sources_nc
from genex_eirene_coupling import main
from temperature_mapping_utils import (
    default_single_species_pseudo_temperature_handler,
    default_sparse_temperature_handler,
)
from CheckedRunEirene import CheckedRunEirene
import os, signal, psutil
import time
import subprocess
import cProfile
import pstats
import dask
import traceback
import sys
import pdb

dask.config.set(scheduler='synchronous')

def test_full_coupling(tmp_path, override, fake=-1):
    from types import SimpleNamespace
    print("=== PYTHON ENTRY REACHED ===", flush=True)
    pid = int(Path("genex.pid").read_text())
    if fake==-1:
        genex_path = "/pscratch/sd/j/jonroelt/D3D_184833_t4800ms_try10/"
    elif fake==0:
        genex_path = "/pscratch/sd/j/jonroelt/source_testing/3D_source/test_script_wo_genex_my_data"
    else:
        genex_path = "/pscratch/sd/j/jonroelt/source_testing/3D_source/full_workflow_test"
    eirene_path = "/pscratch/sd/j/jonroelt/source_testing/3D_source/full_workflow_test/eirene_setup_files"

    checked_runner = CheckedRunEirene()
    print(f"Using file from genex_path: {genex_path}", flush=True)
    try:
        args = SimpleNamespace(
            pid=pid,
            MAX_TIMEOUTS=1,
            eirene_time=1,
            SumTemp=True,
            filepattern=tmp_path + "input_sources_",
            genex_time_index_override=override,
            genex_path=genex_path,
            eirene_path=eirene_path,
            eirene_command="pwd",
        )

        deps = SimpleNamespace(
            run_eirene=checked_runner,
            write_nc=checked_write_nc,   # wrapped
            killpg=lambda pid, sig: cancel_slurm_job(),
            sleep=lambda x: time.sleep(0.1),
            replace=os.replace,
            pid_exists=psutil.pid_exists,
            collision_mappers=None,
            sparse_temperature_handler=default_sparse_temperature_handler,
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

def checked_write_nc(filename, interp_sources, dim_RZ):
    import numpy as np

    # --- check BEFORE writing ---
    for field, species_dict in interp_sources.items():
        for sp, arr in species_dict.items():
            arr_np = np.asarray(arr["SUM"])
            assert np.isfinite(arr_np).all(), \
                f"NaNs in interpolated sources ({field}, {sp})"

            assert not np.all(arr_np == 0), \
                f"Degenerate zero field ({field}, {sp})"

    # call real writer
    write_sources_nc(filename, interp_sources, dim_RZ)

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
