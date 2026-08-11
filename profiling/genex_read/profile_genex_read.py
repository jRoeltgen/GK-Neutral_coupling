#!/usr/bin/env python3
"""Profile GENE-X field loading using completed diagnostic files."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import platform
import pstats
import resource
import sys
import threading
import time

import dask
import numpy as np
import psutil

from neutral_coupling.genex_coupling import genex_interface


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genex-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("legacy", "averaged", "full"),
                        default="averaged")
    parser.add_argument("--scope", choices=("read", "interpolate"),
                        default="read")
    parser.add_argument("--eirene-path", type=Path)
    parser.add_argument("--time-index", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cprofile", type=Path)
    parser.add_argument("--sample-interval", type=float, default=0.05)
    return parser.parse_args()


class MemorySampler:
    def __init__(self, interval):
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        process = psutil.Process()
        start = time.perf_counter()
        while not self.stop.wait(self.interval):
            self.samples.append({
                "seconds": time.perf_counter() - start,
                "rss_bytes": process.memory_info().rss,
            })

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def checksum_fields(fields):
    digest = hashlib.sha256()
    summaries = {}
    for field in sorted(fields):
        for species_name in sorted(fields[field]):
            values = np.ascontiguousarray(np.asarray(fields[field][species_name]))
            digest.update(field.encode())
            digest.update(species_name.encode())
            digest.update(values.view(np.uint8))
            summaries[f"{field}/{species_name}"] = {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "finite_sum": float(np.nansum(values)),
            }
    return digest.hexdigest(), summaries


def legacy_materialize(fields):
    """Reproduce the former duplicate Dask computation for comparison only."""
    for block in fields.values():
        for value in block.values():
            if dask.is_dask_collection(value.data):
                value.data.compute()
    return ORIGINAL_MATERIALIZE(fields)


def load_once(args, grid, equi, params, norm, species, interpolation=None):
    mode = "averaged" if args.mode == "legacy" else args.mode
    if args.mode == "legacy":
        genex_interface.materialize_fields = legacy_materialize
    try:
        fields, tau = genex_interface.load_latest_genex_fields(
            args.genex_path, species, grid, equi, params, norm,
            args.time_index, timeout=30, read_mode=mode,
            read_attempts=1,
        )
        if interpolation is not None:
            coupling, gmtry, tri, pol_mask, rad_mask = interpolation
            coupling.unnormalize_all(fields)
            fields = coupling.interpolate_all_moments(
                gmtry, tri, fields, pol_mask, rad_mask
            )
        return fields, tau
    finally:
        genex_interface.materialize_fields = ORIGINAL_MATERIALIZE


ORIGINAL_MATERIALIZE = genex_interface.materialize_fields


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dask.config.set(scheduler="synchronous")

    init_start = time.perf_counter()
    grid, equi, params, norm, r_all, z_all, compute = genex_interface.wait_for_genex_init(
        args.genex_path
    )
    names = params.get("params_species", {}).get("names", [])
    params.get("params_species", {})["names"] = [name.strip() for name in names]
    species = genex_interface.get_genex_species(params)
    interpolation = None
    if args.scope == "interpolate":
        if args.eirene_path is None:
            raise ValueError("--eirene-path is required for interpolation profiling")
        from neutral_coupling.genex_coupling import genex_eirene_coupling as coupling

        _, b2dat, pol_mask, rad_mask = coupling.eirene_interface(
            args.eirene_path, args.eirene_path
        )
        grid_r = np.asarray(r_all * norm["R0"])
        grid_z = np.asarray(z_all * norm["R0"])
        tri = coupling.build_triangulation(grid_r[compute], grid_z[compute])
        interpolation = (coupling, b2dat.gmtry, tri, pol_mask, rad_mask)
    init_seconds = time.perf_counter() - init_start

    profiler = cProfile.Profile() if args.cprofile else None
    iterations = []
    with MemorySampler(args.sample_interval) as memory:
        if profiler:
            profiler.enable()
        for number in range(args.iterations):
            started = time.perf_counter()
            fields, tau = load_once(
                args, grid, equi, params, norm, species, interpolation
            )
            elapsed = time.perf_counter() - started
            checksum, summaries = checksum_fields(fields)
            iterations.append({
                "iteration": number,
                "seconds": elapsed,
                "tau": float(tau),
                "checksum": checksum,
                "fields": summaries,
            })
            del fields
        if profiler:
            profiler.disable()

    if profiler:
        profiler.dump_stats(args.cprofile)
        with args.cprofile.with_suffix(".txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(50)

    paths = genex_interface.get_mom_paths(args.genex_path)
    report = {
        "mode": args.mode,
        "scope": args.scope,
        "genex_path": str(args.genex_path),
        "time_index": args.time_index,
        "initialization_seconds": init_seconds,
        "diagnostics": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in paths
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dask": dask.__version__,
        },
        "iterations": iterations,
        "memory_samples": memory.samples,
        "peak_sampled_rss_bytes": max(
            (sample["rss_bytes"] for sample in memory.samples), default=0
        ),
        "process_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
