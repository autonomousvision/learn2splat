#!/usr/bin/env bash
# _common/env_setup.sh
# Source this file to set up the environment on the cluster.
# When run locally (no SLURM_JOB_ID), cluster-specific setup is skipped.

echo "Starting script at $(date)"

# Reduce CUDA allocator fragmentation (safe default for our eval workloads).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "SLURM job: $SLURM_JOB_ID  node: $(hostname)"

    source ~/.bashrc

    # Conda environment to run in: either a name ("learn2splat") or a full prefix path
    # ("/path/to/.conda/envs/learn2splat"). Set CONDA_ENV in your environment or .env.
    CONDA_ENV="${CONDA_ENV:-learn2splat}"
    echo "Activating conda environment: ${CONDA_ENV}"
    conda activate "${CONDA_ENV}"
else
    echo "Running locally (no SLURM_JOB_ID) — skipping cluster env setup."
    echo "Assuming conda environment is already active: $(conda info --envs 2>/dev/null | grep '*' || echo 'unknown')"
fi
