"""Characterization tests for the checkpoint/CLI config merging in learn2splat.config.

These pin the *current* merge behavior (test-vs-train strategies and the
checkpoint-source priority) so the merge helpers can be refactored safely. They
exercise the helpers directly with plain OmegaConf configs and temp config
files, so no Hydra runtime is needed..


_merge_train_mode — checkpoint wins, CLI fills gaps

  1. test_checkpoint_wins_cli_fills_missing — with overlapping configs: CLI-only fields survive, shared fields take the checkpoint value (top-level and nested), and
  checkpoint-only fields appear. Confirms the "trained architecture is preserved, new fields filled from CLI" contract.
  2. test_orig_cli_is_pre_merge_snapshot — the returned orig_cli snapshot holds the original CLI values, not the merged ones (this is what _apply_cli_overrides later relies on to
  restore explicit CLI overrides).

  _merge_test_mode — CLI is the base, architecture patched from checkpoint

  3. test_full_model_patches_optimizer_decoder_initializer — with pretrained_model: CLI-base fields (dataset.roots) are untouched; scene_optimizer + decoder take checkpoint
  values with CLI filling missing fields (lr kept); and because a full model bundles initializer weights, the initializer arch is also patched from the checkpoint.
  4. test_optimizer_only_keeps_cli_initializer — with pretrained_optimizer only: optimizer/decoder are patched from the checkpoint, but the CLI initializer is kept (an optimizer
  checkpoint carries no initializer weights). This is the key behavioral difference from case 3.

  _resolve_config_paths — which config file loads, and source priority

  5. test_pretrained_model_resolves_sibling_config — resolves the sibling config.yaml next to the checkpoint dir.
  6. test_model_wins_over_optimizer — when both are set, pretrained_model wins (priority order).
  7. test_optimizer_plus_initializer_sets_both_paths — pretrained_optimizer sets the main config_path, and pretrained_initializer separately sets initializer_config_path.
  8. test_train_mode_without_load_existing_cfg_loads_nothing — in train mode without load_existing_cfg, should_load is false → no config loaded even though a checkpoint is given.
  9. test_resume_with_load_existing_cfg_uses_output_dir_config — resume + load_existing_cfg overrides the path to the output dir's config.yaml.

  What it deliberately does not cover

  - _apply_cli_overrides (the ARCH_KEYS/CLI_WINS_SUBKEYS re-application logic) — I flagged that as the part to leave alone; it isn't exercised here.
  - The full merge_config_from_file orchestration end-to-end (needs HydraConfig), HF-ref resolution, config migration, and wandb-flatten — these characterize the strategy
  helpers, not the file I/O around them.

"""
from pathlib import Path

from omegaconf import OmegaConf

from learn2splat.config import (
    _merge_train_mode,
    _merge_test_mode,
    _resolve_config_paths,
)


def _ckpt_with_config(tmp_path: Path, name: str, config_body: dict) -> Path:
    """Create <tmp>/<name>/checkpoints/model.ckpt and a sibling config.yaml.

    Returns the .ckpt path. _find_config_for_checkpoint resolves the config as
    ckpt.parent.parent / config.yaml, i.e. the run dir alongside checkpoints/.
    """
    run_dir = tmp_path / name
    ckpt = run_dir / "checkpoints" / "model.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("")  # contents irrelevant; only the path is used
    OmegaConf.save(OmegaConf.create(config_body), run_dir / "config.yaml")
    return ckpt


def _cli_cfg(**checkpointing):
    """Minimal CLI config for _resolve_config_paths with checkpointing overrides."""
    base = dict(
        pretrained_model=None,
        pretrained_optimizer=None,
        pretrained_initializer=None,
        resume=False,
        load_existing_cfg=False,
    )
    base.update(checkpointing)
    return OmegaConf.create({"mode": "test", "output_dir": "out", "checkpointing": base})


# --------------------------------------------------------------------------- #
# _merge_train_mode: checkpoint wins for existing fields, CLI fills new fields
# --------------------------------------------------------------------------- #
class TestMergeTrainMode:
    def test_checkpoint_wins_cli_fills_missing(self):
        cli = OmegaConf.create({
            "a": 1,                       # cli-only -> kept
            "shared": "cli",              # in both  -> checkpoint wins
            "scene_trainer": {"x": "cli", "cli_only": 7},
        })
        loaded = OmegaConf.create({
            "shared": "ckpt",
            "scene_trainer": {"x": "ckpt", "ckpt_only": 9},
        })

        merged, orig_cli = _merge_train_mode(cli, loaded, initializer_config_path=None)

        assert merged.a == 1                       # cli-only field preserved
        assert merged.shared == "ckpt"             # checkpoint wins
        assert merged.scene_trainer.x == "ckpt"    # checkpoint wins (nested)
        assert merged.scene_trainer.cli_only == 7  # cli fills field absent in ckpt
        assert merged.scene_trainer.ckpt_only == 9

    def test_orig_cli_is_pre_merge_snapshot(self):
        cli = OmegaConf.create({"shared": "cli", "scene_trainer": {"x": "cli"}})
        loaded = OmegaConf.create({"shared": "ckpt", "scene_trainer": {"x": "ckpt"}})

        _, orig_cli = _merge_train_mode(cli, loaded, initializer_config_path=None)

        # The snapshot keeps the original CLI values, not the checkpoint-merged ones.
        assert orig_cli.shared == "cli"
        assert orig_cli.scene_trainer.x == "cli"


# --------------------------------------------------------------------------- #
# _merge_test_mode: CLI is the base; optimizer/decoder/initializer architecture
# is patched in from the checkpoint (checkpoint wins, CLI fills missing fields).
# --------------------------------------------------------------------------- #
class TestMergeTestMode:
    def _cfgs(self, pretrained_model=None, pretrained_optimizer=None):
        cli = OmegaConf.create({
            "mode": "test",
            "checkpointing": {
                "pretrained_model": pretrained_model,
                "pretrained_optimizer": pretrained_optimizer,
            },
            "dataset": {"roots": "cli_dataset"},       # cli-only base, untouched
            "scene_trainer": {
                "scene_optimizer": {"name": "cli_opt", "lr": 0.1},
                "decoder": {"name": "cli_dec"},
                "scene_initializer": {"name": "cli_init"},
            },
        })
        loaded = OmegaConf.create({
            "scene_trainer": {
                "scene_optimizer": {"name": "ckpt_opt", "extra": 5},
                "decoder": {"name": "ckpt_dec"},
                "scene_initializer": {"name": "ckpt_init"},
            },
        })
        return cli, loaded

    def test_full_model_patches_optimizer_decoder_initializer(self):
        cli, loaded = self._cfgs(pretrained_model="some.ckpt")

        merged, _ = _merge_test_mode(
            cli, loaded, initializer_config_path=None, pretrained_initializer=None
        )

        # dataset (CLI base) is untouched.
        assert merged.dataset.roots == "cli_dataset"
        # Optimizer/decoder architecture comes from the checkpoint; CLI fills missing.
        assert merged.scene_trainer.scene_optimizer.name == "ckpt_opt"
        assert merged.scene_trainer.scene_optimizer.lr == 0.1     # cli-only field kept
        assert merged.scene_trainer.scene_optimizer.extra == 5
        assert merged.scene_trainer.decoder.name == "ckpt_dec"
        # A full pretrained_model bundles initializer weights -> its arch is patched too.
        assert merged.scene_trainer.scene_initializer.name == "ckpt_init"

    def test_optimizer_only_keeps_cli_initializer(self):
        cli, loaded = self._cfgs(pretrained_optimizer="some.ckpt")

        merged, _ = _merge_test_mode(
            cli, loaded, initializer_config_path=None, pretrained_initializer=None
        )

        # pretrained_optimizer carries no initializer weights -> CLI initializer kept.
        assert merged.scene_trainer.scene_initializer.name == "cli_init"
        # Optimizer/decoder are still patched from the checkpoint.
        assert merged.scene_trainer.scene_optimizer.name == "ckpt_opt"
        assert merged.scene_trainer.decoder.name == "ckpt_dec"


# --------------------------------------------------------------------------- #
# _resolve_config_paths: which config file is loaded, and the source priority.
# --------------------------------------------------------------------------- #
class TestResolveConfigPaths:
    def test_pretrained_model_resolves_sibling_config(self, tmp_path):
        ckpt = _ckpt_with_config(tmp_path, "model_run", {"k": "v"})
        cli = _cli_cfg(pretrained_model=str(ckpt))

        config_path, init_path = _resolve_config_paths(cli)

        assert config_path == ckpt.parent.parent / "config.yaml"
        assert init_path is None

    def test_model_wins_over_optimizer(self, tmp_path):
        model_ckpt = _ckpt_with_config(tmp_path, "model_run", {"k": "model"})
        opt_ckpt = _ckpt_with_config(tmp_path, "opt_run", {"k": "opt"})
        cli = _cli_cfg(pretrained_model=str(model_ckpt), pretrained_optimizer=str(opt_ckpt))

        config_path, _ = _resolve_config_paths(cli)

        assert config_path == model_ckpt.parent.parent / "config.yaml"

    def test_optimizer_plus_initializer_sets_both_paths(self, tmp_path):
        opt_ckpt = _ckpt_with_config(tmp_path, "opt_run", {"k": "opt"})
        init_ckpt = _ckpt_with_config(tmp_path, "init_run", {"k": "init"})
        cli = _cli_cfg(
            pretrained_optimizer=str(opt_ckpt), pretrained_initializer=str(init_ckpt)
        )

        config_path, init_path = _resolve_config_paths(cli)

        assert config_path == opt_ckpt.parent.parent / "config.yaml"
        assert init_path == init_ckpt.parent.parent / "config.yaml"

    def test_train_mode_without_load_existing_cfg_loads_nothing(self, tmp_path):
        ckpt = _ckpt_with_config(tmp_path, "model_run", {"k": "v"})
        cli = _cli_cfg(pretrained_model=str(ckpt))
        cli.mode = "train"  # should_load = (test) or load_existing_cfg -> False here

        config_path, init_path = _resolve_config_paths(cli)

        assert config_path is None
        assert init_path is None

    def test_resume_with_load_existing_cfg_uses_output_dir_config(self, tmp_path):
        (tmp_path / "config.yaml").write_text("k: v\n")
        cli = _cli_cfg(resume=True, load_existing_cfg=True)
        cli.mode = "train"
        cli.output_dir = str(tmp_path)

        config_path, _ = _resolve_config_paths(cli)

        assert config_path == tmp_path / "config.yaml"
