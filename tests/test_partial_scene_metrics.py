"""Validates the partial-scene metric reduction (the L321 audit).

When a scene stops early its per-step metric row is shorter than the rest. Two independent
reductions must agree: MetaTrainer._reduce_partial_metric (on_test_end, torch) and
scripts/testing/aggregate_metrics.py (numpy). Both must NaN-flag any step not every scene
reached and report the per-step scene count.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Three scenes; scene 2 stopped early (2 of 4 steps). Steps 0-1 reached by all 3 -> finite mean;
# steps 2-3 reached by only 2 -> NaN. Counts: [3, 3, 2, 2].
ROWS = [[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0], [30.0, 31.0]]
EXPECTED_MEAN = [20.0, 21.0, math.nan, math.nan]
EXPECTED_COUNTS = [3, 3, 2, 2]


def _assert_mean(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        if math.isnan(e):
            assert math.isnan(a), f"expected NaN, got {a}"
        else:
            assert a == pytest.approx(e), f"expected {e}, got {a}"


def test_reduce_partial_metric_nan_flag_and_counts():
    from learn2splat.meta_trainer.meta_trainer import MetaTrainer

    matrix, per_step_mean, scenes_per_step = MetaTrainer._reduce_partial_metric(ROWS)

    _assert_mean(per_step_mean, EXPECTED_MEAN)
    assert scenes_per_step == EXPECTED_COUNTS
    # The padded matrix keeps every scene's row, padding the short one with NaN.
    assert len(matrix) == 3 and all(len(r) == 4 for r in matrix)
    assert math.isnan(matrix[2][2]) and math.isnan(matrix[2][3])


def test_reduce_partial_metric_accepts_int_rows():
    """The iterations metric holds ints (iteration numbers), not floats; the reduction must accept them."""
    from learn2splat.meta_trainer.meta_trainer import MetaTrainer

    _, per_step_mean, scenes_per_step = MetaTrainer._reduce_partial_metric([[0, 2, 4], [0, 2, 4], [0, 2]])
    _assert_mean(per_step_mean, [0.0, 2.0, math.nan])
    assert scenes_per_step == [3, 3, 2]


def test_aggregate_metrics_agrees_on_partial(tmp_path):
    """aggregate_metrics.py over per-scene JSONs must match _reduce_partial_metric."""
    opt = "myoptimizer"  # name must contain "optimizer"
    metrics_dir = tmp_path / opt / "metrics"
    for s, row in enumerate(ROWS):
        scene_dir = metrics_dir / f"scene{s}"
        scene_dir.mkdir(parents=True)
        (scene_dir / f"target_{opt}.json").write_text(json.dumps({
            f"{opt}_psnr": row,
            f"{opt}_iterations": list(range(len(row))),
            "peak_vram_mb": 100.0 + s,  # a scalar (non-list) per-scene metric
        }))

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/testing/aggregate_metrics.py"), str(tmp_path)],
        check=True, cwd=REPO_ROOT,
    )

    averaged = json.loads((metrics_dir / "averaged" / f"target_{opt}.json").read_text())

    _assert_mean(averaged[f"{opt}_psnr"], EXPECTED_MEAN)
    assert averaged["num_scenes_per_step"] == EXPECTED_COUNTS
    assert averaged["num_scenes"] == 3
    # Scalar metric is plain-averaged across scenes.
    assert averaged["peak_vram_mb"] == pytest.approx((100 + 101 + 102) / 3)

    # The two implementations agree on the per-step mean.
    _, helper_mean, helper_counts = __import__(
        "learn2splat.meta_trainer.meta_trainer", fromlist=["MetaTrainer"]
    ).MetaTrainer._reduce_partial_metric(ROWS)
    _assert_mean(averaged[f"{opt}_psnr"], helper_mean)
    assert averaged["num_scenes_per_step"] == helper_counts