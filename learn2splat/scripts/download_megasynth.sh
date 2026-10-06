#!/usr/bin/env bash
# Download pre-rendered MegaSynth dataset splits from Hugging Face.
#
# MegaSynth (https://huggingface.co/datasets/hwjiang/MegaSynth) ships its
# pre-rendered RGB-D scenes as 100 zip files, split_0.zip .. split_99.zip, each
# ~40 GB and holding ~670 scenes. Use this instead of generating scenes locally
# with the MegaSynth/Blender pipeline. Downloaded splits can then be packed into
# the training-ready .torch format with learn2splat/scripts/convert_megasynth.py.
#
# Usage:
#   learn2splat/scripts/download_megasynth.sh <output_dir> [splits] [--extract]
#
#   <output_dir>  where split_N.zip (and extracted/ if --extract) are written
#   [splits]      which splits to fetch: a single "N", a range "A-B",
#                 a comma list "A,B,C", or "all" (default: 0)
#   --extract     unzip each split into <output_dir>/extracted/ after download
#
# Examples:
#   learn2splat/scripts/download_megasynth.sh /data/megasynth 0
#   learn2splat/scripts/download_megasynth.sh /data/megasynth 0-3 --extract
#   learn2splat/scripts/download_megasynth.sh /data/megasynth all
#
# Notes:
#   * Each split is ~40 GB; "all" is ~4 TB -- check free space first.
#   * Downloads are resumable: re-running continues a partial file (curl -C -).
#   * Only curl + unzip are required (no Python / huggingface_hub needed).

set -euo pipefail

REPO_URL="https://huggingface.co/datasets/hwjiang/MegaSynth/resolve/main"

if [ $# -lt 1 ]; then
  echo "usage: $0 <output_dir> [splits|all] [--extract]" >&2
  exit 1
fi
OUTPUT_DIR="$1"; shift

# Remaining args: an optional splits spec and/or the --extract flag, any order.
SPLITS_SPEC="0"
EXTRACT=0
for arg in "$@"; do
  case "$arg" in
    --extract) EXTRACT=1 ;;
    *)         SPLITS_SPEC="$arg" ;;
  esac
done

# Expand the splits spec ("all" | "A-B" | "A,B,C" | "N") into a list of indices.
if [ "$SPLITS_SPEC" = "all" ]; then
  SPLITS=$(seq 0 99)
elif [[ "$SPLITS_SPEC" == *-* ]]; then
  SPLITS=$(seq "${SPLITS_SPEC%-*}" "${SPLITS_SPEC#*-}")
else
  SPLITS="${SPLITS_SPEC//,/ }"
fi

mkdir -p "$OUTPUT_DIR"

for i in $SPLITS; do
  fname="split_${i}.zip"
  url="${REPO_URL}/${fname}"
  dst="${OUTPUT_DIR}/${fname}"

  echo "==> downloading ${fname}"
  curl -L -C - -o "$dst" "$url"

  # Verify against the size Hugging Face reports for the LFS object.
  expected=$(curl -sI "$url" | tr -d '\r' | awk -F': ' 'tolower($1)=="x-linked-size"{print $2}' || true)
  actual=$(stat -c%s "$dst")
  if [ -n "$expected" ] && [ "$expected" != "$actual" ]; then
    echo "    WARNING: size mismatch for ${fname}: got ${actual}, expected ${expected}" >&2
  else
    echo "    size OK (${actual} bytes)"
  fi

  if [ "$EXTRACT" -eq 1 ]; then
    echo "    extracting -> ${OUTPUT_DIR}/extracted/"
    unzip -q -o "$dst" -d "${OUTPUT_DIR}/extracted"
  fi
done

echo "done."
