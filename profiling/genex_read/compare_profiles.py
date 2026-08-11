#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+", type=Path)
parser.add_argument("--require-equal", action="store_true")
args = parser.parse_args()

print(f"{'mode':<10} {'scope':<12} {'iterations':>10} {'mean_s':>12} {'peak_GiB':>12} checksum")
all_checksums = []
for path in args.reports:
    report = json.loads(path.read_text())
    runs = report["iterations"]
    mean = sum(run["seconds"] for run in runs) / len(runs)
    peak = report["peak_sampled_rss_bytes"] / 2**30
    checksums = {run["checksum"] for run in runs}
    checksum = next(iter(checksums)) if len(checksums) == 1 else "VARIES"
    all_checksums.append(checksum)
    print(f"{report['mode']:<10} {report.get('scope', 'read'):<12} "
          f"{len(runs):>10} {mean:>12.3f} {peak:>12.3f} {checksum}")

if args.require_equal and len(set(all_checksums)) != 1:
    raise SystemExit("Field checksums differ between read modes")
