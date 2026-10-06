"""Hydra-free checkpoint -> optimizer construction.

Rebuilds the learned optimizer (architecture + weights) from a checkpoint
*without* going through Hydra. Only ``_load_checkpoint_cfg`` +
``load_typed_config`` are used (both Hydra-free); the Hydra coupling lives in
``setup_cfg`` / ``merge_config_from_file`` / ``setup_output_dir`` which we never
call. All heavy imports are deferred into the functions so ``import learn2splat``
stays cheap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from learn2splat.experimental.api.integration.scene_protocol import Learn2SplatError


@lru_cache(maxsize=8)
def _load_ckpt_cfg_cached(cfg_path_str: str):
    """Load + migrate a checkpoint config once per path (read-only callers).

    ``build_optimizer_cfg`` / ``build_decoder`` / ``get_scene_trainer_scalar``
    all need the same DictConfig; caching avoids re-parsing the file.
    """
    from learn2splat.config import _load_checkpoint_cfg  # Hydra-free

    return _load_checkpoint_cfg(Path(cfg_path_str))


def get_scene_trainer_scalar(cfg_path: Path, key: str, default):
    """Read ``scene_trainer.<key>`` from a checkpoint config (or ``default``).

    Used for scalars that live on the (Hydra-free unavailable) scene-trainer
    config rather than the optimizer cfg: ``num_update_steps``,
    ``iter_batch_size``, ``sh_degree_interval``.
    """
    from omegaconf import OmegaConf

    cfg = _load_ckpt_cfg_cached(str(cfg_path))
    return OmegaConf.select(cfg, f"scene_trainer.{key}", default=default)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

    from learn2splat.scene_trainer.initializer import ResplatInitializerCfg
    from learn2splat.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizerCfg


def resolve_checkpoint_and_config(checkpoint: str, *, what: str) -> tuple[str, Path]:
    """Resolve a checkpoint ref to ``(local_ckpt_path, sibling config.yaml path)``.

    Shared by the Learn2Splat / Learn2SplatInitializer facades. ``checkpoint`` is an
    ``hf://org/repo/file`` reference or a local path; ``what`` names the
    architecture being rebuilt (used only in the error message). Raises
    ``Learn2SplatError`` if no sibling ``config.yaml`` can be found.
    """
    from learn2splat.config import _find_config_for_checkpoint, resplat_official_config_path
    from learn2splat.misc.hf_ckpt import hf_sibling_config, maybe_resolve_hf_ref

    local_ckpt = maybe_resolve_hf_ref(checkpoint)
    cfg_path = (
        hf_sibling_config(checkpoint)
        or _find_config_for_checkpoint(local_ckpt)
        or resplat_official_config_path(local_ckpt)
    )
    if cfg_path is None:
        raise Learn2SplatError(
            f"no config.yaml found next to checkpoint {local_ckpt!r} (looked for "
            f"<ckpt>/../../config.yaml and the wandb latest-run fallback). Learn2Splat "
            f"needs the training config to rebuild the {what}."
        )
    return local_ckpt, cfg_path


def _optimizer_class_by_cfg_name():
    """Map a checkpoint config's ``scene_optimizer.name`` -> optimizer class.

    A checkpoint config's ``scene_optimizer.name`` is the cfg literal
    (``"knn_based"`` / ``"l2s"`` / ``"resplat_v1"`` / ``"resplat_v2"``), and each
    class asserts ``cfg.name`` equals its ``OPTIMIZER_NAME``.
    """
    from learn2splat.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizer
    from learn2splat.scene_trainer.optimizer.optimizer_learn2splat import (
        Learn2SplatOptimizer,
    )
    from learn2splat.scene_trainer.optimizer.optimizer_resplat import (
        ResplatOptimizerV1,
        ResplatOptimizerV2,
    )

    classes = (
        KnnBasedOptimizer,
        Learn2SplatOptimizer,
        ResplatOptimizerV1,
        ResplatOptimizerV2,
    )
    return {cls.OPTIMIZER_NAME: cls for cls in classes}


def _initializer_cfg_class(name: str):
    """Map ``scene_initializer.name`` -> its concrete typed Cfg dataclass.

    ``InitializerCfg`` is a PEP-604 union; dacite needs a concrete dataclass
    as the top-level target (a union is only resolvable as a *field* type).
    Keyed to match both ``SCENE_INITIALIZERS`` and each Cfg's ``name``
    Literal.
    """
    from learn2splat.scene_trainer.initializer import (
        InitializerColmapCfg,
        InitializerEdgsCfg,
        InitializerPlyCfg,
        InitializerPointcloudCfg,
        InitializerRandomCfg,
        ResplatInitializerCfg,
    )

    return {
        "resplat_v1": ResplatInitializerCfg,
        "resplat_v2": ResplatInitializerCfg,
        "colmap": InitializerColmapCfg,
        "ply": InitializerPlyCfg,
        "edgs": InitializerEdgsCfg,
        "random": InitializerRandomCfg,
        "pointcloud": InitializerPointcloudCfg,
    }.get(name)


def _compose_default_group(group: str, value: str):
    """Hydra-compose the bundled default for ``scene_trainer.<group>=<value>``.

    Released checkpoints predate fields later added to the typed configs
    (e.g. ``scene_optimizer.refiner.fallback_means_lr``). The training/eval
    pipeline reconciles this by merging the checkpoint config over the
    *current* default config (config.py:merge_config_from_file). We mirror
    that: compose the bundled default for the group (e.g.
    ``scene_optimizer=knn_based`` -> base -> refiner:none, or
    ``scene_initializer=colmap``) so missing fields can be backfilled with
    current defaults while checkpoint values win for shared keys.

    Scoped use of ``hydra.compose`` (no ``@hydra.main`` / ``HydraConfig.get``,
    no app context); lazily imported so ``import learn2splat`` stays light. Returns
    ``None`` if composition fails (caller falls back to a strict parse).
    """
    try:
        import learn2splat
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf

        config_dir = str(Path(learn2splat.__file__).resolve().parent / "config")
        GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(version_base=None, config_dir=config_dir):
                composed = compose(
                    config_name="main",
                    overrides=[f"scene_trainer/{group}={value}"],
                )
        finally:
            GlobalHydra.instance().clear()
        return OmegaConf.select(composed, f"scene_trainer.{group}")
    except Exception as e:  # noqa: BLE001 - best-effort backfill
        print(
            f"[learn2splat] warning: could not compose default scene_trainer.{group}"
            f"={value} for back-compat merge ({type(e).__name__}: {e}); "
            f"parsing checkpoint config as-is."
        )
        return None


def build_optimizer_cfg(cfg_path: Path) -> tuple["KnnBasedOptimizerCfg", int | None]:
    """Load a checkpoint's saved config and return its typed optimizer cfg.

    Returns ``(KnnBasedOptimizerCfg, num_update_steps)`` where
    ``num_update_steps`` (the per-scene optimization step count) is read from
    ``scene_trainer.num_update_steps`` if present (it is NOT part of the
    optimizer cfg), else ``None``.
    """
    from omegaconf import OmegaConf

    from learn2splat.config import load_typed_config
    from learn2splat.scene_trainer.optimizer.optimizer_knn_based import KnnBasedOptimizerCfg

    cfg = _load_ckpt_cfg_cached(str(cfg_path))  # read_omega_cfg + migrate; NO Hydra
    so = OmegaConf.select(cfg, "scene_trainer.scene_optimizer")
    name = OmegaConf.select(cfg, "scene_trainer.scene_optimizer.name")
    if so is None or name in (None, "none"):
        raise Learn2SplatError(
            f"checkpoint config at {cfg_path} has no learned scene_optimizer "
            f"(scene_trainer.scene_optimizer={name!r}). Learn2Splat needs a learned "
            f"optimizer checkpoint (knn_based / l2s / resplat_v1 / resplat_v2)."
        )
    # Backfill fields a released (older) checkpoint config lacks with the
    # current defaults, then let checkpoint values win for shared keys
    # (mirrors config.py:merge_config_from_file's OmegaConf.merge).
    default_so = _compose_default_group("scene_optimizer", "knn_based")
    if default_so is not None:
        OmegaConf.set_struct(default_so, False)
        merged_so = OmegaConf.merge(default_so, so)
    else:
        merged_so = so
    try:
        opt_cfg = load_typed_config(merged_so, KnnBasedOptimizerCfg)
    except Exception as e:  # dacite/omegaconf errors -> actionable message
        raise Learn2SplatError(
            f"failed to parse scene_optimizer from {cfg_path} into "
            f"KnnBasedOptimizerCfg ({type(e).__name__}: {e})."
        ) from e

    # Mirror SceneTrainerCfg (scene_trainer_cfg.py: scene_optimizer.update(
    # scene_initializer)): wire the checkpoint's initializer cfg into the
    # optimizer cfg so the runtime-only fields init_sh_d / sh_d — absent from
    # every config file — are populated before the optimizer nn.Module is built.
    si = OmegaConf.select(cfg, "scene_trainer.scene_initializer")
    si_name = OmegaConf.select(cfg, "scene_trainer.scene_initializer.name")
    if si is None or si_name in (None, "none"):
        raise Learn2SplatError(
            f"checkpoint config at {cfg_path} has no scene_initializer "
            f"(name={si_name!r}); cannot derive the optimizer's initializer "
            f"settings required to build it."
        )
    init_cls = _initializer_cfg_class(str(si_name))
    if init_cls is None:
        raise Learn2SplatError(
            f"unsupported scene_initializer.name={si_name!r} in {cfg_path}; "
            f"cannot derive the optimizer's initializer settings."
        )
    default_si = _compose_default_group("scene_initializer", str(si_name))
    if default_si is not None:
        OmegaConf.set_struct(default_si, False)
        merged_si = OmegaConf.merge(default_si, si)
    else:
        merged_si = si
    try:
        init_cfg = load_typed_config(merged_si, init_cls)
        opt_cfg.update(init_cfg)  # sets init_sh_d/sh_d
    except Exception as e:
        raise Learn2SplatError(
            f"failed to wire scene_initializer ({si_name!r}) into the "
            f"optimizer cfg from {cfg_path} ({type(e).__name__}: {e})."
        ) from e

    num_update_steps = OmegaConf.select(
        cfg, "scene_trainer.num_update_steps", default=None
    )
    return opt_cfg, num_update_steps


def build_initializer_cfg(cfg_path: Path) -> "ResplatInitializerCfg":
    """Load a checkpoint's saved config and return its typed *initializer* cfg.

    The learned-initializer analogue of ``build_optimizer_cfg`` (same load +
    back-compat-merge path), but returns the initializer cfg standalone — for
    building the learned feed-forward initializer via ``Learn2SplatInitializer``.
    Raises ``Learn2SplatError`` unless the checkpoint was trained with a learned
    initializer (``scene_initializer.name`` ``resplat_v1`` / ``resplat_v2``);
    the colmap-init checkpoints carry no initializer weights.
    """
    from omegaconf import OmegaConf

    from learn2splat.config import load_typed_config

    cfg = _load_ckpt_cfg_cached(str(cfg_path))  # read + migrate (resplat -> resplat_v1); NO Hydra
    si = OmegaConf.select(cfg, "scene_trainer.scene_initializer")
    si_name = OmegaConf.select(cfg, "scene_trainer.scene_initializer.name")
    if si is None or str(si_name) not in ("resplat_v1", "resplat_v2"):
        raise Learn2SplatError(
            f"checkpoint config at {cfg_path} has no learned scene_initializer "
            f"(scene_trainer.scene_initializer.name={si_name!r}). The learned "
            f"'resplat' initializer needs a checkpoint trained with "
            f"scene_initializer=resplat_v1/resplat_v2 — e.g. the dedicated "
            f"'resplat_init' checkpoint (hf://autonomousvision/learn2splat/"
            f"resplat_init/...). The dense/colmap checkpoints carry no "
            f"initializer weights."
        )
    init_cls = _initializer_cfg_class(str(si_name))  # ResplatInitializerCfg
    # Backfill fields a released (older) checkpoint config lacks with the current
    # defaults, then let checkpoint values win (mirrors build_optimizer_cfg).
    default_si = _compose_default_group("scene_initializer", str(si_name))
    if default_si is not None:
        OmegaConf.set_struct(default_si, False)
        merged_si = OmegaConf.merge(default_si, si)
    else:
        merged_si = si
    try:
        return load_typed_config(merged_si, init_cls)
    except Exception as e:  # dacite/omegaconf errors -> actionable message
        raise Learn2SplatError(
            f"failed to parse scene_initializer ({si_name!r}) from {cfg_path} into "
            f"{getattr(init_cls, '__name__', init_cls)} ({type(e).__name__}: {e})."
        ) from e


def build_initializer(init_cfg: "ResplatInitializerCfg") -> "nn.Module":
    """Construct the concrete learned initializer for ``init_cfg`` (no weights)."""
    from learn2splat.scene_trainer.initializer import get_scene_initializer

    return get_scene_initializer(init_cfg)


def load_initializer_state(
    initializer: "nn.Module", ckpt_path: str, strict: bool
) -> None:
    """Load initializer weights from ``ckpt_path`` into ``initializer``.

    Uses the shared ``_load_submodule_state`` (``initializer.*`` keys, legacy
    ``encoder.*`` fallback). Unlike the optimizer, there is no ``state_proj`` drop
    or attribute rename (those are optimizer-only).
    """
    isd = _load_submodule_state(ckpt_path, "initializer.")
    if not isd:
        raise Learn2SplatError(
            f"no initializer weights found in {ckpt_path} (looked for "
            f"'initializer.*' or legacy 'encoder.*' keys). Use a learned-init "
            f"checkpoint such as 'resplat_init'."
        )
    initializer.load_state_dict(isd, strict=strict)


def build_decoder(
    cfg_path: Path, dataset_cfg: object, decoder_overrides: dict | None = None
) -> "nn.Module":
    """Build the renderer the checkpoint was trained with.

    Uses ``scene_trainer.decoder`` from the checkpoint config (NOT a hardcoded
    backend): the learned optimizer's in-loop render gradients must match the
    backend it trained with, and only the registered/available backends are
    usable (e.g. ``gsplat`` — the learn2splat default; the ``inria`` backend needs
    ``diff_gaussian_rasterization``, which is optional). ``dataset_cfg`` only
    needs a ``background_color`` attribute. ``decoder_overrides`` (e.g.
    ``rasterize_mode`` / ``eps2d``) take precedence over the checkpoint config.
    """
    from omegaconf import OmegaConf

    from learn2splat.config import load_typed_config
    from learn2splat.model.decoder import DECODER_CFGS, get_decoder

    cfg = _load_ckpt_cfg_cached(str(cfg_path))
    node = OmegaConf.select(cfg, "scene_trainer.decoder")
    if node is None:
        raise Learn2SplatError(
            f"checkpoint config at {cfg_path} has no scene_trainer.decoder; "
            f"cannot rebuild the renderer the optimizer trained with."
        )
    # Resolve the discriminated union by `name` (dacite can't parse the union
    # itself). A missing name means its optional backend isn't installed.
    name = OmegaConf.select(node, "name")
    cfg_cls = DECODER_CFGS.get(name)
    if cfg_cls is None:
        raise Learn2SplatError(
            f"decoder backend {name!r} (from {cfg_path}) is not available in "
            f"this environment. Install its backend (e.g. "
            f"diff_gaussian_rasterization for 'inria') or use a checkpoint "
            f"trained with the 'gsplat' decoder."
        )
    # gsplat decoder rasterize_mode / eps2d, by precedence:
    #   caller override  >  checkpoint config  >  gsplat rasterization() default
    # (so an older checkpoint that omits a field behaves as plain gsplat would).
    if name == "gsplat":
        import inspect

        from gsplat.rendering import rasterization

        sig = inspect.signature(rasterization).parameters
        node = OmegaConf.merge(
            OmegaConf.create(
                {f: sig[f].default for f in ("rasterize_mode", "eps2d") if f in sig}
            ),
            node,
            OmegaConf.create(dict(decoder_overrides or {})),
        )
    try:
        decoder_cfg = load_typed_config(node, cfg_cls)
    except Exception as e:
        raise Learn2SplatError(
            f"failed to parse scene_trainer.decoder from {cfg_path} "
            f"({type(e).__name__}: {e})."
        ) from e
    return get_decoder(decoder_cfg, dataset_cfg)


def build_optimizer(opt_cfg: "KnnBasedOptimizerCfg") -> "nn.Module":
    """Construct the concrete learned optimizer for ``opt_cfg`` (no weights)."""
    from learn2splat.misc.io import FrequencyScheduler

    mapping = _optimizer_class_by_cfg_name()
    cls = mapping.get(opt_cfg.name)
    if cls is None:
        raise Learn2SplatError(
            f"unsupported scene_optimizer.name={opt_cfg.name!r}; Learn2Splat supports "
            f"{sorted(mapping)}."
        )
    optimizer = cls(opt_cfg)
    # The optimizer's save_every (info/context/target/debug artifact dumps) is
    # wired by SceneTrainer during training; the optimizer calls it
    # unconditionally, so the API inference path — which has nothing to dump —
    # installs a disabled scheduler instead of leaving it None.
    save_every = FrequencyScheduler(last_step=0)
    save_every.disable(True)
    optimizer.save_every = save_every
    return optimizer


def build_refiner_cfg(strategy: str, num_refine: int) -> "BaseStrategyCfg":
    """Load the bundled ``refiner/{strategy}.yaml`` and, when it densifies, rescale
    its schedule so ADC actually fires within ``num_refine`` steps.

    ``strategy`` is a raw refiner name ("none"/"default"/"edgs"/"mcmc"/… — a sibling
    .yaml under ``config/scene_trainer/scene_optimizer/refiner/``). The bundled
    configs use the original 3DGS schedule (densify from iter 500), which never
    triggers in the demo's ~100-step runs. We scale by a *fraction of N clamped to
    the config's own cadence*: ``start``/``every`` = min(10%·N, config value),
    ``stop`` = min(50%·N, config value). For full-length runs the clamp wins and the
    native schedule is recovered exactly (fastgs @ N=30000: start/every=500,
    stop=15000 — ~28 densify rounds); for short demo runs the fraction wins so ADC
    still fires a few times with a refine tail (N=100: start/every=10, stop=50 → ~3
    rounds). ``reset_every`` is left at its (absolute) bundled value, so runs shorter
    than it never reset (no thrash) while long runs keep the config cadence.
    ``cap_max`` is bounded for VRAM safety. Returns a ``BaseStrategyCfg``.
    """
    from dataclasses import replace

    import learn2splat
    from omegaconf import OmegaConf

    from learn2splat.config import load_typed_config
    from learn2splat.scene_trainer.adc.fastgs import FastGSStrategyCfg
    from learn2splat.scene_trainer.adc.mcmc import McmcStrategyCfg
    from learn2splat.scene_trainer.adc.vanilla import VanillaStrategyCfg

    cfg_dir = Path(learn2splat.__file__).resolve().parent / "config"
    yaml_path = (
        cfg_dir / "scene_trainer" / "scene_optimizer" / "refiner" / f"{strategy}.yaml"
    )
    if not yaml_path.exists():
        avail = sorted(p.stem for p in yaml_path.parent.glob("*.yaml"))
        raise Learn2SplatError(
            f"unknown ADC strategy {strategy!r}: no refiner config at {yaml_path} "
            f"(available: {', '.join(avail)})."
        )
    # Build the matching StrategyCfg variant (by name) so the strategy-specific fields are present;
    # a bare BaseStrategyCfg would lack noise_lr (mcmc) / grad_abs_thresh (fastgs).
    variant = {"mcmc": McmcStrategyCfg, "fastgs": FastGSStrategyCfg}.get(strategy, VanillaStrategyCfg)
    cfg = load_typed_config(OmegaConf.load(yaml_path), variant)

    # Non-densifying strategies (e.g. "none") leave the Gaussian set fixed — no rescale.
    if not (cfg.do_densify or cfg.do_prune or cfg.do_opacity_reset):
        return cfg

    n = max(1, int(num_refine))
    cap = cfg.cap_max if (cfg.cap_max and cfg.cap_max > 0) else 1_500_000
    # Fraction of N, clamped to the config's native cadence (see docstring): long runs
    # recover the bundled schedule exactly, short runs compress but still densify.
    return replace(
        cfg,
        refine_start_iter=min(max(1, round(0.1 * n)), cfg.refine_start_iter),
        refine_every=min(max(1, round(0.1 * n)), cfg.refine_every),
        refine_stop_iter=min(round(0.5 * n), cfg.refine_stop_iter),
        cap_max=cap,
    )


def build_adam_baseline(
    num_refine: int, adc: str = "none", base: str = "adam"
) -> "nn.Module":
    """Build the codebase's 3DGS Adam optimizer for a fair baseline comparison.

    Uses the bundled ``scene_optimizer={base}`` config (default ``adam`` —
    gsplat's example hyperparameters; ``fastgs`` swaps in FastGS's LRs) for the
    optimizer hyperparameters (LRs, betas), with the means-LR decay horizon set to
    ``num_refine``. ``adc`` selects the densification strategy: the default
    ``"none"`` disables ADC so the baseline refines the same fixed Gaussian set as
    the learned optimizer (a like-for-like update-rule comparison); any other raw
    refiner name swaps in that strategy's schedule (scaled to ``num_refine`` via
    ``build_refiner_cfg``). Returns a ready-to-run ``AdamOptimizer``.
    """
    from dataclasses import asdict

    from omegaconf import OmegaConf

    from learn2splat.config import load_typed_config
    from learn2splat.misc.io import FrequencyScheduler
    from learn2splat.scene_trainer.optimizer.optimizer_adam import (
        AdamOptimizer,
        AdamOptimizerCfg,
    )

    composed = _compose_default_group("scene_optimizer", base)
    if composed is None:
        raise Learn2SplatError(
            f"could not Hydra-compose the bundled 'scene_optimizer={base}' config "
            "for the Adam baseline."
        )
    OmegaConf.set_struct(composed, False)
    # gsplat decays the means LR over the full step budget.
    composed.lr_scheduler.max_steps = int(num_refine)
    if adc == "none":
        # Disable densification — the baseline refines the same fixed Gaussian set
        # as the learned optimizer (a like-for-like comparison of the update rule).
        for flag in ("do_densify", "do_prune", "do_opacity_reset"):
            if flag in composed.refiner:
                composed.refiner[flag] = False
    else:
        # Swap in the chosen ADC strategy (schedule scaled to num_refine).
        composed.refiner = asdict(build_refiner_cfg(adc, num_refine))
    try:
        adam_cfg = load_typed_config(composed, AdamOptimizerCfg)
    except Exception as e:
        raise Learn2SplatError(
            f"failed to parse the bundled '{base}' config into AdamOptimizerCfg "
            f"({type(e).__name__}: {e})."
        ) from e

    optimizer = AdamOptimizer(adam_cfg)
    save_every = FrequencyScheduler(last_step=0)  # nothing to dump (see build_optimizer)
    save_every.disable(True)
    optimizer.save_every = save_every
    # AdamOptimizer is a NonlearnedOptimizer — already pinned to eval mode.
    return optimizer


def _load_submodule_state(ckpt_path: str, prefix: str) -> dict:
    """Load ``ckpt_path`` and return the sub-state-dict under ``prefix``.

    Strips the Lightning ``scene_trainer.`` prefix, then takes the unified
    ``<prefix>*`` keys (e.g. ``optimizer.`` / ``initializer.``) or, failing that,
    the legacy ``encoder.*`` keys (pre init/opt split). Shared by
    ``load_optimizer_state`` / ``load_initializer_state``.
    """
    import torch

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("scene_trainer.", ""): v for k, v in state.items()}
    if any(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}


def load_optimizer_state(
    optimizer: "nn.Module",
    ckpt_path: str,
    init_state_wo_features: bool,
    strict: bool,
) -> None:
    """Load optimizer weights from ``ckpt_path`` into ``optimizer``.

    Transcribes the prefix-stripping / legacy-rename / feature-drop logic from
    ``learn2splat/main.py:load_optimizer`` (we cannot call that function: it needs a
    full Hydra ``cfg`` and a ``scene_trainer``).
    """
    from learn2splat.misc.checkpointing import _rename_optimizer_attrs

    # Rename module attributes renamed in the update_/refine_ cleanup pass
    # (and the resplat-era render_error_mv_attn).
    osd = _rename_optimizer_attrs(_load_submodule_state(ckpt_path, "optimizer."))

    if not osd:
        raise Learn2SplatError(
            f"no optimizer weights found in {ckpt_path} (looked for "
            f"'optimizer.*' or legacy 'encoder.*' keys)."
        )

    if init_state_wo_features:
        osd = {k: v for k, v in osd.items() if "state_proj" not in k}

    optimizer.load_state_dict(osd, strict=strict)
