"""Compute per-image rendering diffs between two output directories.

Pairs PNGs by relative path under each root (e.g. .../initializerply/images/<scene>/color_target/*.png)
and reports max|diff|, mean|diff|, PSNR. Useful for comparing the gsplat and inria decoders
on the same init.

Usage:
    python -m learn2splat.scripts.diff_renders <root_a> <root_b> [--subdir initializerply/images] \
        [--save-diff <out_dir>] [--top-k 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def collect_pngs(root: Path, subdir: str) -> dict[str, Path]:
    base = root / subdir if subdir else root
    if not base.exists():
        sys.exit(f"Missing path: {base}")
    return {str(p.relative_to(base)): p for p in base.rglob("*.png")}


def diff_pair(a_path: Path, b_path: Path) -> dict:
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        return {"shape_a": a.shape, "shape_b": b.shape, "skipped": True}
    d = np.abs(a - b)
    mse = float((d ** 2).mean())
    psnr = float(20 * np.log10(1.0) - 10 * np.log10(mse + 1e-12))
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "mse": mse,
        "psnr": psnr,
        "shape": list(a.shape),
    }


def save_diff_image(a_path: Path, b_path: Path, out_path: Path, scale: float = 5.0) -> None:
    a = np.asarray(Image.open(a_path).convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(Image.open(b_path).convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        return
    d = np.clip(np.abs(a - b) * scale, 0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((d * 255).astype(np.uint8)).save(out_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root_a", type=Path)
    p.add_argument("root_b", type=Path)
    p.add_argument("--subdir", default="",
                   help="Restrict comparison to this subpath under each root (e.g. 'initializerply/images').")
    p.add_argument("--save-diff", type=Path, default=None,
                   help="Directory to save scaled |a-b| images, mirroring the relative path.")
    p.add_argument("--diff-scale", type=float, default=5.0)
    p.add_argument("--top-k", type=int, default=10, help="Show K worst pairs by max|diff|.")
    p.add_argument("--json", type=Path, default=None, help="Optional path to dump per-pair stats as JSON.")
    p.add_argument("--pair", choices=["name", "sorted"], default="name",
                   help="'name': pair by matching relative path. 'sorted': pair by index after sorting "
                        "each root's PNGs (use when filename schemes differ but order corresponds).")
    args = p.parse_args()

    pngs_a = collect_pngs(args.root_a, args.subdir)
    pngs_b = collect_pngs(args.root_b, args.subdir)

    print(f"root_a: {args.root_a / args.subdir if args.subdir else args.root_a}")
    print(f"root_b: {args.root_b / args.subdir if args.subdir else args.root_b}")

    if args.pair == "name":
        common = sorted(set(pngs_a) & set(pngs_b))
        only_a = sorted(set(pngs_a) - set(pngs_b))
        only_b = sorted(set(pngs_b) - set(pngs_a))
        print(f"pair=name; common: {len(common)}; only_a: {len(only_a)}; only_b: {len(only_b)}")
        if not common:
            sys.exit("No common PNGs to diff. Try --pair sorted if filenames differ but order matches.")
        pairs = [(rel, pngs_a[rel], pngs_b[rel]) for rel in common]
    else:
        sa = sorted(pngs_a.items())
        sb = sorted(pngs_b.items())
        if len(sa) != len(sb):
            sys.exit(f"pair=sorted: counts differ (root_a={len(sa)}, root_b={len(sb)}); can't pair by index.")
        print(f"pair=sorted; {len(sa)} pairs")
        only_a = only_b = []
        pairs = [(f"{ra}|{rb}", pa, pb) for (ra, pa), (rb, pb) in zip(sa, sb)]

    results = []
    skipped = []
    for rel, a_path, b_path in pairs:
        stats = diff_pair(a_path, b_path)
        if stats.get("skipped"):
            skipped.append((rel, stats))
            continue
        results.append((rel, stats))
        if args.save_diff is not None:
            save_diff_image(a_path, b_path, args.save_diff / rel.replace("|", "_VS_"), scale=args.diff_scale)

    if skipped:
        print(f"\nShape-mismatch pairs ({len(skipped)}):")
        for rel, s in skipped[:10]:
            print(f"  {rel}: {s['shape_a']} vs {s['shape_b']}")

    if not results:
        sys.exit("All pairs had mismatched shapes.")

    max_abs = np.array([s["max_abs"] for _, s in results])
    mean_abs = np.array([s["mean_abs"] for _, s in results])
    psnr = np.array([s["psnr"] for _, s in results])

    print(f"\nPer-pair stats ({len(results)} pairs):")
    print(f"  max|diff|  — min: {max_abs.min():.4e}  median: {np.median(max_abs):.4e}  max: {max_abs.max():.4e}")
    print(f"  mean|diff| — min: {mean_abs.min():.4e}  median: {np.median(mean_abs):.4e}  max: {mean_abs.max():.4e}")
    print(f"  PSNR(dB)   — min: {psnr.min():.2f}    median: {np.median(psnr):.2f}    max: {psnr.max():.2f}")

    results.sort(key=lambda r: -r[1]["max_abs"])
    print(f"\nWorst {min(args.top_k, len(results))} pairs by max|diff|:")
    for rel, s in results[: args.top_k]:
        print(f"  max={s['max_abs']:.4e}  mean={s['mean_abs']:.4e}  psnr={s['psnr']:.2f}dB  {rel}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(
                {
                    "root_a": str(args.root_a),
                    "root_b": str(args.root_b),
                    "subdir": args.subdir,
                    "common_count": len(common),
                    "only_a": only_a,
                    "only_b": only_b,
                    "pairs": [{"rel": r, **s} for r, s in results],
                    "skipped": [{"rel": r, **s} for r, s in skipped],
                },
                f,
                indent=2,
            )
        print(f"\nWrote per-pair stats to {args.json}")


if __name__ == "__main__":
    main()
