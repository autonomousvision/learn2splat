"""Aggregate per-scene metrics into averaged results.

Usage: python scripts/testing/aggregate_metrics.py <output_dir>

Finds the subdirectory of <output_dir> whose name contains 'optimizer',
then aggregates all per-scene JSON files under <optimizer_dir>/metrics/.

Example:
    python scripts/testing/aggregate_metrics.py \
        results/dl3dv/8_8_low_res_140_scenes/resplat_v1_init/joint/2000_150000_speedup
"""
import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `learn2splat` is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from learn2splat.misc.io import CustomPath

output_dir = CustomPath(sys.argv[1])

# Find the subdirectory whose name contains "optimizer"
optimizer_dirs = [d for d in output_dir.iterdir() if d.is_dir() and "optimizer" in d.name.lower()]

if len(optimizer_dirs) == 0:
    print(f"ERROR: No subdirectory containing 'optimizer' found in {output_dir}")
    sys.exit(1)
if len(optimizer_dirs) > 1:
    print(f"WARNING: Multiple optimizer dirs found, using the first: {optimizer_dirs}")

metrics_dir = optimizer_dirs[0] / "metrics"

if not metrics_dir.is_dir():
    print(f"ERROR: metrics dir not found: {metrics_dir}")
    sys.exit(1)

print(f"Aggregating metrics from: {metrics_dir}")

results = {}
scene_count = 0

for scene_dir in sorted(metrics_dir.iterdir()):
    if not scene_dir.is_dir() or scene_dir.name == "averaged":
        continue

    json_files = list(scene_dir.glob("*.json"))
    if not json_files:
        print(f"Skipping {scene_dir.name}: no JSON files")
        continue

    for json_file in json_files:
        with json_file.open() as f:
            data = json.load(f)
        file_results = results.setdefault(json_file.name, {})
        for key, val in data.items():
            file_results.setdefault(key, []).append(val)

    scene_count += 1

print(f"Aggregated {scene_count} scenes")

out_dir = metrics_dir / "averaged"
out_dir.mkdir(exist_ok=True)

for filename, file_results in results.items():
    averaged = {"num_scenes": scene_count}  # always first
    for key, values in file_results.items():
        if values and isinstance(values[0], list):
            # Per-step metric. A scene that stopped early (partial optimization) has fewer
            # steps, so the rows are ragged: NaN-pad to the longest. The mean then goes NaN at
            # any step not every scene reached, flagging it as not comparable (rather than
            # silently averaging the survivors). (No-op when every scene completed.)
            max_steps = max(len(v) for v in values)
            padded = np.full((len(values), max_steps), np.nan)
            for r, v in enumerate(values):
                padded[r, :len(v)] = v
            averaged[key] = np.mean(padded, axis=0).tolist()
            # Scenes contributing to each step (same across per-step metrics, so it is
            # written once); records the population behind any NaN-flagged step.
            averaged["num_scenes_per_step"] = (~np.isnan(padded)).sum(axis=0).tolist()
        else:
            # Scalar per-scene metric (e.g. timing / peak VRAM).
            averaged[key] = np.mean(np.array(values), axis=0).tolist()

    with (out_dir / filename).open("w") as f:
        json.dump(averaged, f, indent=4)

print(f"Wrote averaged results ({scene_count} scenes) to {out_dir}/")

# Now also update output_dir/metrics directory to have a copy of the averaged results for easier access
# And create a separate json file for each averaged metric, e.g. output_dir/metrics/{filename.stem}_{key}.json
(output_dir / "metrics").mkdir(parents=True, exist_ok=True)
for filename in results.keys():
    averaged_file = out_dir / filename
    if not averaged_file.is_file():
        print(f"ERROR: Averaged file not found: {averaged_file}")
        continue

    with averaged_file.open() as f:
        averaged_data = json.load(f)

    # Copy the whole averaged file to output_dir/metrics
    (output_dir / "metrics" / filename).write_text(json.dumps(averaged_data, indent=4))

    # Create separate JSON files for each metric
    for key, value in averaged_data.items():
        # remove optimizer prefix if present so it won't appear twice
        metric = key.replace(f"{optimizer_dirs[0].name}_", "")
        metric_filename = f"{CustomPath(filename).stem}_{metric}.json"
        # Write only the value of this metric to a separate JSON file (no key, just the value)
        (output_dir / "metrics" / metric_filename).write_text(json.dumps(value, indent=4))
