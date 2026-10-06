"""
Show results table for a set of scenes from a results directory.

Single-run mode — shows metrics across all iterations:
    python scripts/show_results_table.py <run_dir>

Comparison mode — pass multiple run dirs, picks one iteration per run:
    python scripts/show_results_table.py <run_dir_A> <run_dir_B> [--iter 2000] [--labels A B]

In comparison mode --iter N selects the closest available iteration <= N in each
run (default: last iteration).  Rows are scenes, columns are runs.

Layout auto-detection (applied per run):

  Layout A (multi-scene): scenes are subdirectories inside the run directory.
    {run_dir}/{optimizer}/metrics/{scene}/target_{opt}.json

  Layout B (per-scene siblings): scenes are sibling directories at dataset level.
    {dataset_dir}/{scene}/{suffix}/metrics/target_{opt}.json   (combined)
    {dataset_dir}/{scene}/{suffix}/metrics/target_{opt}_psnr.json  (separate)

--scene-depth N  (Layout B only) levels above run_dir to the scene directory (default: 3).

python scripts/show_results_table.py \
        results/mipnerf360/8_all_9_scenes/dense_colmap_init/joint/2000_150000_speedup \
        results/mipnerf360/8_all_9_scenes/dense_colmap_init/adam_tuned/2000 \
        --labels learn2splat adam_tuned
"""

import argparse
import json
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

SCENE_W_MAX = 12  # truncate long scene names (e.g. DL3DV 64-char hashes) for display


def short_scene(name: str, width: int = SCENE_W_MAX) -> str:
    """Trim an over-wide scene name with an ellipsis (display only; keys stay full)."""
    return name if len(name) <= width else name[: width - 1] + "…"


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_iter(itr: float) -> str:
    itr = int(itr)
    if itr == 0:
        return "0"
    if itr % 1000 == 0:
        return f"{itr // 1000}K"
    return str(itr)


def ms_to_min(ms: float) -> float:
    return ms / 60_000


def fmt_val(v, fmt=".3f", scale=1.0) -> str:
    if v is None:
        return "n/a"
    return format(v * scale, fmt)


def pick_iter_index(iterations: list, target: int | None) -> int:
    """Return index of the closest iteration <= target, or last if target is None."""
    if target is None:
        return len(iterations) - 1
    best = 0
    for i, it in enumerate(iterations):
        if it <= target:
            best = i
    return best


def auto_label(path: Path) -> str:
    """Derive a short label from the last two path components."""
    parts = path.resolve().parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


# ── data loading ─────────────────────────────────────────────────────────────

METRIC_FILES = {
    "psnr":       "target_{opt}_psnr.json",
    "ssim":       "target_{opt}_ssim.json",
    "lpips":      "target_{opt}_vgg_lpips.json",
    "time_ms":    "target_{opt}_time.json",
    "gaussians":  "target_{opt}_gaussians.json",
    "iterations": "target_{opt}_iterations.json",
}

COMBINED_KEYS = {
    "psnr":       "{opt}_psnr",
    "ssim":       "{opt}_ssim",
    "lpips":      "{opt}_vgg_lpips",
    "time_ms":    "{opt}_time",
    "gaussians":  "{opt}_gaussians",
    "iterations": "{opt}_iterations",
}


def detect_optimizer_name(metrics_dir: Path) -> str | None:
    for f in metrics_dir.glob("target_*_psnr.json"):
        return f.stem[len("target_"):][:-len("_psnr")]
    for f in metrics_dir.glob("target_*.json"):
        data = load_json(f)
        if isinstance(data, dict) and any(k.endswith("_psnr") for k in data):
            return f.stem[len("target_"):]
    return None


def load_scene_metrics(metrics_dir: Path) -> dict | None:
    """Load metrics from a per-scene metrics directory. Returns None if not found."""
    if not metrics_dir.is_dir():
        return None

    opt = detect_optimizer_name(metrics_dir)
    if opt is None:
        return None

    combined_path = metrics_dir / f"target_{opt}.json"
    if combined_path.exists():
        data = load_json(combined_path)
        if not isinstance(data, dict):
            return None

        def read_combined(key):
            v = data.get(COMBINED_KEYS[key].format(opt=opt))
            if v is None:
                return None
            return v[0] if isinstance(v[0], list) else v

        iters = read_combined("iterations")
        if iters is None:
            return None
        return {
            "iterations": iters,
            "psnr":       read_combined("psnr"),
            "ssim":       read_combined("ssim"),
            "lpips":      read_combined("lpips"),
            "time_ms":    read_combined("time_ms"),
            "gaussians":  read_combined("gaussians"),
        }

    def read_separate(key):
        d = load_json(metrics_dir / METRIC_FILES[key].format(opt=opt))
        if d is None:
            return None
        return d[0] if isinstance(d[0], list) else d

    iters = read_separate("iterations")
    if iters is None:
        return None
    return {
        "iterations": iters,
        "psnr":       read_separate("psnr"),
        "ssim":       read_separate("ssim"),
        "lpips":      read_separate("lpips"),
        "time_ms":    read_separate("time_ms"),
        "gaussians":  read_separate("gaussians"),
    }


def find_all_run_dirs(root: Path) -> list[Path]:
    """Find all run dirs under root that contain *optimizer/metrics/{scene}/target_*.json."""
    run_dirs = set()
    for json_file in root.glob("**/*optimizer/metrics/*/target_*optimizer.json"):
        if json_file.parent.name == "averaged":
            continue
        run_dirs.add(json_file.parent.parent.parent.parent)  # parent of *optimizer dir
    return sorted(run_dirs)


def find_multi_scene_metrics(run_dir: Path) -> tuple[list[str], dict]:
    """Layout A: {run_dir}/{optimizer}/metrics/{scene}/target_{opt}.json"""
    for opt_dir in sorted(run_dir.iterdir()):
        if not opt_dir.is_dir() or opt_dir.name in ("metrics", "averaged"):
            continue
        nested_metrics = opt_dir / "metrics"
        if not nested_metrics.is_dir():
            continue
        candidates = []
        for scene_dir in sorted(nested_metrics.iterdir()):
            if scene_dir.is_dir() and scene_dir.name != "averaged":
                metrics = load_scene_metrics(scene_dir)
                if metrics is not None:
                    candidates.append((scene_dir.name, metrics))
        if candidates:
            return [n for n, _ in candidates], {n: m for n, m in candidates}
    return [], {}


def load_run(run_path: Path, scene_depth: int) -> tuple[list[str], dict]:
    """Load scenes+metrics for one run directory (auto-detects layout)."""
    scenes, scene_metrics = find_multi_scene_metrics(run_path)
    if scenes:
        return scenes, scene_metrics

    # Layout B: sibling scene directories
    scene_dir = run_path
    for _ in range(scene_depth):
        scene_dir = scene_dir.parent
    dataset_dir = scene_dir.parent
    suffix = run_path.relative_to(scene_dir)

    scenes = []
    scene_metrics = {}
    for sd in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        metrics = load_scene_metrics(sd / suffix / "metrics")
        if metrics is not None:
            scenes.append(sd.name)
            scene_metrics[sd.name] = metrics
        else:
            print(f"  [skip] {sd.name}: no metrics at {sd / suffix / 'metrics'}")
    return scenes, scene_metrics


def enrich(scene_metrics: dict) -> None:
    """Add time_min and gaussians_m derived fields in-place."""
    for m in scene_metrics.values():
        tm = m.get("time_ms")
        gs = m.get("gaussians")
        m["time_min"] = [ms_to_min(v) for v in tm] if tm else None
        m["gaussians_m"] = [v / 1e6 for v in gs] if gs else None


# ── single-run tables ─────────────────────────────────────────────────────────

def print_metric_table(title: str, iters: list[str], scenes: list[str],
                       data: dict, fmt=".3f", scale=1.0):
    """Rows=iterations, columns=scenes+Mean."""
    disp = [short_scene(s) for s in scenes]
    col_w = max(10, max(len(d) for d in disp))
    iter_w = 8
    sep = "-" * (iter_w + 2 + (col_w + 2) * len(scenes) + col_w + 2 + 6)
    print(f"\n{'─' * 4} {title} {'─' * max(0, len(sep) - len(title) - 6)}")
    header = f"{'Iter':<{iter_w}}"
    for d in disp:
        header += f"  {d:>{col_w}}"
    header += f"  {'Mean':>{col_w}}"
    print(header)
    print(sep)
    for i, it_label in enumerate(iters):
        row_vals = [data.get(s)[i] if (data.get(s) is not None and i < len(data[s])) else None
                    for s in scenes]
        valid = [v * scale for v in row_vals if v is not None]
        mean = sum(valid) / len(valid) if valid else None
        row = f"{it_label:<{iter_w}}"
        for v in row_vals:
            row += f"  {fmt_val(v, fmt, scale):>{col_w}}"
        row += f"  {fmt_val(mean, fmt, 1.0):>{col_w}}"
        print(row)


def print_summary_table(iters: list[str], scenes: list[str], all_data: dict,
                         compress: bool = False):
    """Rows=iterations, columns=metrics (means across scenes)."""
    metrics_cfg = [
        ("psnr",        "PSNR",          ".3f", 1.0),
        ("ssim",        "SSIM",          ".4f", 1.0),
        ("lpips",       "LPIPS",         ".4f", 1.0),
        ("time_min",    "Time (min)",    ".2f", 1.0),
        ("gaussians_m", "Gaussians (M)", ".3f", 1.0),
    ]
    if compress:
        metrics_cfg = [m for m in metrics_cfg if m[0] in ("psnr", "time_min")]
    col_w = 14
    iter_w = 8
    header = f"{'Iter':<{iter_w}}" + "".join(f"  {lbl:>{col_w}}" for _, lbl, _, _ in metrics_cfg)
    sep = "-" * len(header)
    print(f"\n{'─' * 4} Summary (mean across {len(scenes)} scenes) {'─' * max(0, len(sep) - 40)}")
    print(header)
    print(sep)
    for i, it_label in enumerate(iters):
        row = f"{it_label:<{iter_w}}"
        for key, _, fmt, scale in metrics_cfg:
            vals = [all_data[s][key][i] for s in scenes
                    if all_data.get(s, {}).get(key) is not None and i < len(all_data[s][key])]
            mean = sum(vals) / len(vals) if vals else None
            row += f"  {fmt_val(mean, fmt, scale):>{col_w}}"
        print(row)


# ── comparison tables ─────────────────────────────────────────────────────────

def print_compare_metric_table(title: str, all_scenes: list[str], runs: list[str],
                                run_data: dict, metric: str, fmt=".3f", scale=1.0):
    """Rows=scenes+Mean, columns=runs.  run_data[run][scene] = scalar value or None."""
    col_w = max(12, max(len(r) for r in runs))
    n_total = len(all_scenes)

    # Pre-compute per-run counts so the mean label width is known before printing sep
    run_means: dict[str, list] = {r: [] for r in runs}
    scene_rows: list[tuple[str, list]] = []
    for scene in all_scenes:
        vals = []
        for r in runs:
            v = run_data[r].get(scene)
            if v is not None:
                run_means[r].append(v * scale)
            vals.append(v)
        scene_rows.append((scene, vals))

    counts = [len(run_means[r]) for r in runs]
    min_count = min(counts) if counts else 0
    if min_count < n_total:
        mean_label = f"Mean ({min_count}/{n_total})"
    else:
        mean_label = f"Mean ({n_total})"

    scene_w = max(10, max(len(short_scene(s)) for s in all_scenes), len(mean_label))
    sep = "-" * (scene_w + 2 + (col_w + 2) * len(runs))
    print(f"\n{'─' * 4} {title} {'─' * max(0, len(sep) - len(title) - 6)}")
    header = f"{'Scene':<{scene_w}}" + "".join(f"  {r:>{col_w}}" for r in runs)
    print(header)
    print(sep)

    for scene, vals in scene_rows:
        row = f"{short_scene(scene):<{scene_w}}"
        for v in vals:
            row += f"  {fmt_val(v, fmt, scale):>{col_w}}"
        print(row)

    print(sep)
    row = f"{mean_label:<{scene_w}}"
    for r in runs:
        vals = run_means[r]
        mean = sum(vals) / len(vals) if vals else None
        row += f"  {fmt_val(mean, fmt, 1.0):>{col_w}}"
    print(row)


def print_compare_summary_best_psnr(all_scenes: list[str], runs: list[str],
                                     run_scene_full: dict, compress: bool = False):
    """One-line per run: metrics at the iteration with highest mean PSNR across scenes."""
    metrics_cfg = [
        ("psnr",        "PSNR",          ".3f", 1.0),
        ("ssim",        "SSIM",          ".4f", 1.0),
        ("lpips",       "LPIPS",         ".4f", 1.0),
        ("time_min",    "Time (min)",    ".2f", 1.0),
        ("gaussians_m", "Gaussians (M)", ".3f", 1.0),
    ]
    if compress:
        metrics_cfg = [m for m in metrics_cfg if m[0] in ("psnr", "time_min")]
    run_w = max(12, max(len(r) for r in runs))
    iter_w = 8
    n_w = 7
    col_w = 14
    header = (f"{'Run':<{run_w}}  {'@BestIter':<{iter_w}}  {'Scenes':>{n_w}}"
              + "".join(f"  {lbl:>{col_w}}" for _, lbl, _, _ in metrics_cfg))
    sep = "-" * len(header)
    print(f"\n{'─' * 4} Best-PSNR summary (best mean-PSNR iteration) {'─' * max(0, len(sep) - 50)}")
    print(header)
    print(sep)
    n_total = len(all_scenes)
    for r in runs:
        run_data = run_scene_full.get(r, {})
        scenes_present = [s for s in all_scenes if run_data.get(s, {}).get("psnr")]
        if not scenes_present:
            continue

        # Use iterations from the first available scene (assumed shared across scenes)
        iterations = run_data[scenes_present[0]]["iterations"]

        # Find the iteration index with the highest mean PSNR across scenes
        best_idx, best_mean = 0, -float("inf")
        for i in range(len(iterations)):
            vals = [run_data[s]["psnr"][i] for s in scenes_present
                    if i < len(run_data[s]["psnr"])]
            if vals and (m := sum(vals) / len(vals)) > best_mean:
                best_mean, best_idx = m, i

        n_scenes = len(scenes_present)
        n_label = f"{n_scenes}/{n_total}" if n_scenes < n_total else str(n_scenes)
        row = f"{r:<{run_w}}  {fmt_iter(iterations[best_idx]):<{iter_w}}  {n_label:>{n_w}}"
        for key, _, fmt, scale in metrics_cfg:
            vals = [run_data[s][key][best_idx] for s in scenes_present
                    if run_data[s].get(key) and best_idx < len(run_data[s][key])]
            mean = sum(vals) / len(vals) if vals else None
            row += f"  {fmt_val(mean, fmt, scale):>{col_w}}"
        print(row)


def print_compare_summary(all_scenes: list[str], runs: list[str],
                           run_scene_vals: dict, iter_labels: dict,
                           compress: bool = False):
    """One-line per run: mean PSNR / SSIM / LPIPS / Time / Gaussians + iteration used."""
    metrics_cfg = [
        ("psnr",        "PSNR",          ".3f", 1.0),
        ("ssim",        "SSIM",          ".4f", 1.0),
        ("lpips",       "LPIPS",         ".4f", 1.0),
        ("time_min",    "Time (min)",    ".2f", 1.0),
        ("gaussians_m", "Gaussians (M)", ".3f", 1.0),
    ]
    if compress:
        metrics_cfg = [m for m in metrics_cfg if m[0] in ("psnr", "time_min")]
    run_w = max(12, max(len(r) for r in runs))
    iter_w = 6
    n_w = 7
    col_w = 14
    header = (f"{'Run':<{run_w}}  {'@Iter':<{iter_w}}  {'Scenes':>{n_w}}"
              + "".join(f"  {lbl:>{col_w}}" for _, lbl, _, _ in metrics_cfg))
    sep = "-" * len(header)
    print(f"\n{'─' * 4} Comparison summary (mean across scenes) {'─' * max(0, len(sep) - 44)}")
    print(header)
    print(sep)
    n_total = len(all_scenes)
    for r in runs:
        scene_vals_for_run = [v for s in all_scenes
                              if (v := run_scene_vals[r].get(s, {}).get("psnr")) is not None]
        n_scenes = len(scene_vals_for_run)
        n_label = f"{n_scenes}/{n_total}" if n_scenes < n_total else str(n_scenes)
        row = f"{r:<{run_w}}  {iter_labels.get(r, '?'):<{iter_w}}  {n_label:>{n_w}}"
        for key, _, fmt, scale in metrics_cfg:
            vals = [v for s in all_scenes
                    if (v := run_scene_vals[r].get(s, {}).get(key)) is not None]
            mean = sum(vals) / len(vals) if vals else None
            row += f"  {fmt_val(mean, fmt, scale):>{col_w}}"
        print(row)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_paths", type=Path, nargs="+",
                        help="One or more run directories.")
    parser.add_argument("--scene-depth", type=int, default=3,
                        help="(Layout B only) levels above run_dir to scene dir (default: 3).")
    parser.add_argument("--iter", type=int, nargs="+", default=None, dest="target_iter",
                        help="(Comparison mode) pick closest iteration <= N. One value applies to all runs; one per run applies individually. Default: last.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="(Comparison mode) short labels for each run (same order).")
    parser.add_argument("--compress", action="store_true",
                        help="Compressed view: show only PSNR and Time (min) metrics.")
    parser.add_argument("--intersection", action="store_true",
                        help="(Comparison mode) restrict to scenes present in ALL runs, so every "
                             "Mean is computed over the shared scene set (default: union of scenes).")
    args = parser.parse_args()

    # ── single-run mode ───────────────────────────────────────────────────────
    if len(args.run_paths) == 1:
        run_path = args.run_paths[0].resolve()
        if not run_path.is_dir():
            print(f"Error: {run_path} is not a directory.")
            return 1

        scenes, scene_metrics = find_multi_scene_metrics(run_path)
        if not scenes:
            # Auto-discover: treat the given path as a root containing multiple runs
            sub_runs = find_all_run_dirs(run_path)
            if not sub_runs:
                print("No scenes found with metrics. Check the path and --scene-depth.")
                return 1
            print(f"Auto-discovered {len(sub_runs)} run(s) under {run_path}")
            args.run_paths = sub_runs
            if not (args.labels and len(args.labels) == len(sub_runs)):
                args.labels = [str(p.relative_to(run_path)) for p in sub_runs]
            # Fall through to comparison mode below
        else:
            enrich(scene_metrics)
            sample = next(iter(scene_metrics.values()))
            iter_labels = [fmt_iter(v) for v in sample["iterations"]]

            print(f"Run dir    : {run_path}")
            print(f"Scenes     : {', '.join(short_scene(s) for s in scenes)}")
            print(f"Iterations : {', '.join(iter_labels)}")

            def scene_vals(key):
                return {s: scene_metrics[s].get(key) for s in scenes}

            print_metric_table("PSNR (dB)",     iter_labels, scenes, scene_vals("psnr"),        fmt=".3f")
            if not args.compress:
                print_metric_table("SSIM",          iter_labels, scenes, scene_vals("ssim"),         fmt=".4f")
                print_metric_table("LPIPS (VGG)",   iter_labels, scenes, scene_vals("lpips"),        fmt=".4f")
            print_metric_table("Time (min)",    iter_labels, scenes, scene_vals("time_min"),     fmt=".2f")
            if not args.compress:
                print_metric_table("Gaussians (M)", iter_labels, scenes, scene_vals("gaussians_m"),  fmt=".3f")
            print_summary_table(iter_labels, scenes, scene_metrics, compress=args.compress)
            return 0

    # ── comparison mode ───────────────────────────────────────────────────────
    if args.labels and len(args.labels) != len(args.run_paths):
        print(f"Error: --labels count ({len(args.labels)}) != run count ({len(args.run_paths)}).")
        return 1

    labels = args.labels or [auto_label(p) for p in args.run_paths]
    target_iters = args.target_iter if args.target_iter is not None else [None]
    multi_iter = len(target_iters) > 1

    # col_key = "{label}@{iter}" when multiple iters, else just "{label}".
    # run_scene_vals[col_key][scene][metric] = scalar at chosen iteration
    # run_scene_full[label][scene]           = full timeseries (for best-PSNR summary, one per run)
    run_scene_vals: dict[str, dict[str, dict]] = {}
    run_scene_full: dict[str, dict[str, dict]] = {}
    chosen_iter_labels: dict[str, str] = {}
    all_scenes_sets: list[set] = []
    cols: list[str] = []  # ordered column/row keys

    for path, label in zip(args.run_paths, labels):
        path = path.resolve()
        if not path.is_dir():
            print(f"Error: {path} is not a directory.")
            return 1
        print(f"Loading {label} ...")
        scenes, scene_metrics = load_run(path, args.scene_depth)
        if not scenes:
            print(f"  [warn] no scenes found for {label}, skipping.")
            continue
        enrich(scene_metrics)
        all_scenes_sets.append(set(scenes))
        run_scene_full[label] = dict(scene_metrics)

        sample = next(iter(scene_metrics.values()))
        for target in target_iters:
            idx = pick_iter_index(sample["iterations"], target)
            chosen_iter = sample["iterations"][idx]
            col_key = f"{label}@{fmt_iter(chosen_iter)}" if multi_iter else label
            if col_key in run_scene_vals:
                continue
            chosen_iter_labels[col_key] = fmt_iter(chosen_iter)
            cols.append(col_key)

            run_scene_vals[col_key] = {}
            for scene, m in scene_metrics.items():
                run_scene_vals[col_key][scene] = {
                    key: m[key][idx] if (m.get(key) is not None and idx < len(m[key])) else None
                    for key in ("psnr", "ssim", "lpips", "time_min", "gaussians_m")
                }

    if not run_scene_vals:
        print("No runs loaded.")
        return 1

    if args.intersection:
        all_scenes = sorted(set.intersection(*all_scenes_sets))
    else:
        all_scenes = sorted(set.union(*all_scenes_sets))
    runs = [l for l in labels if l in run_scene_full]

    if multi_iter:
        print(f"\nCols    : {', '.join(cols)}")
    else:
        print(f"\nRuns    : {', '.join(f'{c}@{chosen_iter_labels[c]}' for c in cols)}")
    print(f"Scenes  : {', '.join(short_scene(s) for s in all_scenes)}")

    metrics_cfg = [
        ("psnr",        "PSNR (dB)",     ".3f", 1.0),
        ("ssim",        "SSIM",          ".4f", 1.0),
        ("lpips",       "LPIPS (VGG)",   ".4f", 1.0),
        ("time_min",    "Time (min)",    ".2f", 1.0),
        ("gaussians_m", "Gaussians (M)", ".3f", 1.0),
    ]
    if args.compress:
        metrics_cfg = [m for m in metrics_cfg if m[0] in ("psnr", "time_min")]
    for key, title, fmt, scale in metrics_cfg:
        per_col = {c: {s: run_scene_vals[c].get(s, {}).get(key) for s in all_scenes}
                   for c in cols}
        print_compare_metric_table(title, all_scenes, cols, per_col, key, fmt=fmt, scale=scale)

    print_compare_summary(all_scenes, cols, run_scene_vals, chosen_iter_labels,
                          compress=args.compress)
    print_compare_summary_best_psnr(all_scenes, runs, run_scene_full,
                                    compress=args.compress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
