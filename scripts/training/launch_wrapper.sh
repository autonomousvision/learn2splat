#!/bin/bash
#set -e

# Usage check
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <base_command_file> [--resume]"
    exit 1
fi

BASE_CMD_FILE="$1"
RESUME_FLAG="${2:-}"

# Load bashrc and activate conda
source ~/.bashrc

# Conda environment to run in: either a name ("learn2splat") or a full prefix path
# ("/path/to/.conda/envs/learn2splat"). Set CONDA_ENV in your environment or .env.
CONDA_ENV="${CONDA_ENV:-learn2splat}"
echo "Activating conda environment: ${CONDA_ENV}"
conda activate "${CONDA_ENV}"

# Load .env file if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo ""
echo "========================================================================================"
echo "Job ID is $SLURM_JOB_ID"
echo "hostname is $(hostname)"
echo "Work directory is $WORK"
echo "Python version: $(python --version)"
echo "Conda environment: $(conda info --envs | grep '*' | awk '{print $2}')"
echo "Current working directory: $(pwd)"
echo "========================================================================================"
echo ""

# Read command from file (strip empty lines and comments, remove line breaks)
BASE_CMD=$(grep -vE '^\s*#|^\s*$' "$BASE_CMD_FILE" | tr -d '\\\n')
BASE_CMD="$BASE_CMD log_slurm_id=true"

echo ""
echo "Base command:"
echo "$BASE_CMD"
echo ""

MODIFIED_CMD="$BASE_CMD"

# Resume is opt-in via --resume. It enables checkpoint resume (Lightning
# restores weights/optimizer/step from the latest ckpt in output_dir, and
# wandb continues the run found in output_dir/wandb) and drops the
# pretrained_model/pretrained_depth init args, which would otherwise load over
# the restored checkpoint.
if [ "$RESUME_FLAG" = "--resume" ]; then
    echo "Resume mode is enabled."
    MODIFIED_CMD="$MODIFIED_CMD checkpointing.resume=true"
    MODIFIED_CMD=$(echo "$MODIFIED_CMD" | sed 's/checkpointing.pretrained_depth=[^ ]*//g')
    MODIFIED_CMD=$(echo "$MODIFIED_CMD" | sed 's/checkpointing.pretrained_model=[^ ]*//g')
    echo "Modified command (--resume):"
    echo "$MODIFIED_CMD"
fi

# Run the command
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
eval "$MODIFIED_CMD"