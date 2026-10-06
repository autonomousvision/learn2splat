import os
from pathlib import Path

import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from omegaconf import OmegaConf

from learn2splat.misc.LocalLogger import LocalLogger
from learn2splat.paths import DEBUG


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_checkpoint(
    run_id: str,
    download_dir: Path,
    version: str | None,
) -> Path:
    api = wandb.Api()
    run = api.run(run_id)

    # Find the latest saved model checkpoint.
    chosen = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        # If no version is specified, use the latest.
        if version is None:
            if chosen is None or version_to_int(artifact) > version_to_int(chosen):
                chosen = artifact

        # If a specific verison is specified, look for it.
        elif version == artifact.version:
            chosen = artifact
            break

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
    chosen.download(root=root)
    return root / "model.ckpt"


def setup_wandb_logger(cfg, cfg_dict) -> WandbLogger | LocalLogger:
    if cfg_dict.wandb.mode == "disabled" or cfg.mode != "train":
        return LocalLogger()

    wandb_extra_kwargs = {}

    # Detect the wandb id job run if resuming
    if cfg_dict.checkpointing.resume:
        if cfg_dict.wandb.id is None:
            print(f"Resuming wandb run without id, using latest run in output directory.")
            # Find the latest wandb run id in the output directory
            wandb_dir = cfg.output_dir / "wandb" / "latest-run"
            # look for a file name in the format "run-######.wandb" file and extract the id
            wandb_files = list(wandb_dir.glob("run-*.wandb"))
            assert len(wandb_files) <= 1, "Multiple wandb files found in the latest run directory."
            if len(wandb_files) == 1:
                wandb_file = wandb_files[0]
                wandb_id = wandb_file.stem.split('-')[1]
                wandb_extra_kwargs.update({'id': wandb_id, 'resume': "must"})

    if cfg_dict.wandb.id is not None:
        print(f"Setting wandb run with id from cfg {cfg_dict.wandb.id}.")
        wandb_extra_kwargs.update({'id': cfg_dict.wandb.id, 'resume': "must"})

    run_name = os.path.basename(cfg.output_dir)

    if cfg_dict.log_slurm_id:
        hostname = os.uname().nodename
        job_id = os.environ.get('SLURM_JOB_ID', "local run: " + hostname)
        run_name += f" ({job_id})"

    # if debugging, add a tag to the run name
    if DEBUG:
        run_name += "  DEBUG"
        cfg_dict.wandb.update({'tags': ['debug']})
    if os.environ.get('WANDB_ENTITY') is not None:
        cfg_dict.wandb.update({'entity': os.environ.get('WANDB_ENTITY')})

    logger = WandbLogger(
        entity=cfg_dict.wandb.entity,
        project=cfg_dict.wandb.project,
        mode=cfg_dict.wandb.mode,
        name=run_name,
        tags=cfg_dict.wandb.get("tags", None),
        log_model=False,
        save_dir=cfg.output_dir,
        config=OmegaConf.to_container(cfg_dict),
        **wandb_extra_kwargs,
    )

    if logger.experiment is not None:
        # Log code
        logger.experiment.log_code("learn2splat")
        # Log notes
        if cfg_dict.wandb.notes is not None:
            logger.experiment.notes = cfg_dict.wandb.notes

    return logger


def update_checkpoint_path(path: str | None, wandb_cfg: dict) -> Path | None:
    if path is None:
        return None

    if not str(path).startswith("wandb://"):
        return Path(path)

    run_id, *version = path[len("wandb://") :].split(":")
    if len(version) == 0:
        version = None
    elif len(version) == 1:
        version = version[0]
    else:
        raise ValueError("Invalid version specifier!")

    project = wandb_cfg["project"]
    return download_checkpoint(
        f"{project}/{run_id}",
        Path("checkpoints"),
        version,
    )
