#!/bin/bash

# For a dry run, run `bash scripts/training/submit_train.slurm`.
# Otherwise, run `sbatch scripts/training/submit_train.slurm`.

#SBATCH --job-name=dl3dv-train-8views
#SBATCH --partition=***   # set your cluster GPU partition(s)
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=448G
#SBATCH --gres=gpu:4

# #SBATCH --exclude=***   # optional: blacklist specific nodes

#SBATCH --output=logs/%j_train.out
#SBATCH --error=logs/%j_train.err

# For debugging purposes, print the current date and time
echo "Starting job ID: $SLURM_JOB_ID at: $(date)"
echo "$SLURM_JOB_ID" > ~/.last_slurm_job_id
pwd

# Parse args: a training script (positional) and an optional --resume flag.
# With --resume, the run continues from the latest checkpoint in output_dir.
DEFAULT_TRAIN_SCRIPT="./scripts/training/learn2splat_joint.sh"
train_script=""
resume_flag=""
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        resume_flag="--resume"
    else
        train_script="$arg"
    fi
done
train_script="${train_script:-$DEFAULT_TRAIN_SCRIPT}"

echo "Using training script: $train_script"
[ -n "$resume_flag" ] && echo "Resume mode requested."

# Launch the job (via bash so it doesn't depend on the wrapper's exec bit)
bash scripts/training/launch_wrapper.sh "$train_script" $resume_flag || {
    echo "Error: Failed to launch the training script."
    exit 1
}