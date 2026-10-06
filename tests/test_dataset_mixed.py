"""Tests for the mixed-dataset multiplexer (`DatasetMixed`) and its config wiring.

Covers the two highest-risk pieces (Hydra `@package` source composition + dacite `dict[str, Union]`
resolution), the single-dataset non-regression, the train-eval redirect, and the mixer mechanics
(1-source delegation, weight normalisation, weighted interleave with child reopen).
"""
import functools
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from dacite import Config as DaciteConfig, from_dict
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from learn2splat.config import TYPE_HOOKS, get_eval_cfg, load_typed_root_config
from learn2splat.dataset import get_dataset
from learn2splat.dataset.data_module import DataLoaderCfg, DataModule
from learn2splat.dataset.data_types import BatchedViews
from learn2splat.dataset.dataset_dl3dv import DatasetDL3DVCfg
from learn2splat.dataset.dataset_megasynth import DatasetMegasynthCfg
from learn2splat.dataset.dataset_mixed import DatasetMixed, MixedDatasetCfg
from learn2splat.dataset.dataset_re10k import DatasetRE10kCfg
from learn2splat.misc.step_tracker import StepTracker
from learn2splat.scene_trainer.initializer import InitializerColmapCfg, InitializerDepthCfg
from learn2splat.scene_trainer.scene_trainer import SceneTrainer

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = str(REPO / "learn2splat" / "config")
MEGASYNTH = (REPO / "datasets" / "megasynth" / "test" / "index.json").exists()
needs_megasynth = pytest.mark.skipif(not MEGASYNTH, reason="datasets/megasynth not present")


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="main", overrides=overrides)


@functools.lru_cache(maxsize=None)
def _megasynth_cfg() -> DatasetMegasynthCfg:
    """Typed megasynth leaf cfg, resolved from its group yaml (no full RootCfg needed). Cached: the
    compose is the slow part and every consumer treats the returned cfg as read-only."""
    d = OmegaConf.to_container(_compose(["dataset=megasynth"]).dataset, resolve=True)
    return from_dict(DatasetMegasynthCfg, d, DaciteConfig(type_hooks=TYPE_HOOKS))


def _mixed(sources, weights, eval_source) -> MixedDatasetCfg:
    return MixedDatasetCfg(name="mixed", sources=sources, weights=weights,
                           eval_source=eval_source, background_color=[0.0, 0.0, 0.0])


def _data_loader_cfg() -> DataLoaderCfg:
    dl = from_dict(DataLoaderCfg, OmegaConf.to_container(_compose(["dataset=megasynth"]).data_loader))
    dl.train.num_workers = 0  # in-process; batch_size 1 since the mixer interleaves heterogeneous sources
    dl.train.batch_size = 1
    return dl


# --------------------------------------------------------------------------- config resolution
def test_mixed_config_resolves_union():
    """@package composition + dacite dict[str, SingleDatasetCfg] union resolution."""
    cfg = _compose(["+experiment=train_mixed", "checkpointing.pretrained_initializer=null"])
    d = load_typed_root_config(cfg).dataset
    assert isinstance(d, MixedDatasetCfg)
    assert set(d.sources) == {"dense", "sparse"}
    assert all(isinstance(s, DatasetDL3DVCfg) for s in d.sources.values())
    # per-source view samplers survive @package rebasing
    assert d.sources["dense"].view_sampler.num_context_views == 64
    assert d.sources["sparse"].view_sampler.num_context_views == 8
    assert not hasattr(d, "view_sampler")  # sampler is per-source, never global
    assert d.eval_source == "dense"  # eval runs the 64-view colmap source (the default initializer)


def test_single_dataset_unaffected():
    """The union addition must not perturb single-dataset resolution."""
    cfg = _compose(["+experiment=test_re10k", "output_dir=/tmp/regr_check"])
    assert isinstance(load_typed_root_config(cfg).dataset, DatasetRE10kCfg)


def test_get_eval_cfg_redirects_to_eval_source():
    """Train-time eval on a mixed cfg redirects to the declared eval source (sparse dl3dv)."""
    cfg = _compose(["+experiment=train_mixed", "checkpointing.pretrained_initializer=null"])
    eval_cfg = get_eval_cfg(cfg)
    assert isinstance(eval_cfg.dataset, DatasetDL3DVCfg)
    assert "dl3dv" in str(eval_cfg.dataset.roots).lower()
    assert eval_cfg.dataset.view_sampler.name == "evaluation"


# --------------------------------------------------------------------------- mixer mechanics
@needs_megasynth
def test_one_source_delegates():
    """val with a single source yields well-formed items from that source: the mixer's val path
    yields the eval-source child's items (tagging each with the source key), so its output shape
    matches the dataset alone."""
    mega = _megasynth_cfg()
    dm = DatasetMixed(_mixed({"a": mega}, {"a": 1.0}, "a"), "val", StepTracker())
    plain = get_dataset(mega, "val", StepTracker())
    mixed_item = next(iter(dm))
    plain_item = next(iter(plain))
    # the mixer adds a "source" tag (its only addition); the underlying item is otherwise the same shape
    assert mixed_item.keys() == {"context", "target", "scene", "source"}
    assert plain_item.keys() == {"context", "target", "scene"}
    assert mixed_item["source"] == "a"  # val delegates to eval_source and tags it
    assert mixed_item["context"]["image"].shape == plain_item["context"]["image"].shape
    assert isinstance(mixed_item["scene"], str)


@needs_megasynth
def test_weights_normalised_to_probs():
    mega = _megasynth_cfg()
    dm = DatasetMixed(_mixed({"a": mega, "b": mega}, {"a": 1.0, "b": 3.0}, "a"), "train", StepTracker())
    assert torch.allclose(dm.probs, torch.tensor([0.25, 0.75], dtype=dm.probs.dtype))


@needs_megasynth
def test_val_stage_builds_only_eval_source():
    """val/test must not build the non-eval children (mirrors no_mix_test_set)."""
    mega = _megasynth_cfg()
    dm = DatasetMixed(_mixed({"a": mega, "b": mega}, {"a": 1.0, "b": 1.0}, "b"), "val", StepTracker())
    assert len(dm.children) == 1
    assert dm.probs is None


@needs_megasynth
def test_weighted_interleave_and_reopen():
    """Draws track the weights, and exhausted children are transparently reopened."""
    mega = _megasynth_cfg()
    dm = DatasetMixed(_mixed({"a": mega, "b": mega}, {"a": 1.0, "b": 3.0}, "a"), "train", StepTracker())

    # Swap in cheap finite children (dicts, so the mixer can tag item["source"]) to (a) count draws
    # via the tag and (b) force the StopIteration reopen path (each child yields fewer than we pull).
    dm.children = [[{} for _ in range(5)], [{} for _ in range(5)]]
    counts = {"a": 0, "b": 0}
    it = iter(dm)
    for _ in range(4000):
        counts[next(it)["source"]] += 1
    assert sum(counts.values()) == 4000  # reopen kept the infinite stream alive past 5 items
    ratio = counts["b"] / counts["a"]
    assert 2.4 < ratio < 3.6  # ~3:1, loose for RNG


@needs_megasynth
def test_datamodule_collates_mixed_stream():
    """Full data path: DataModule -> DataLoader collate -> BatchedViews.from_dict (what the
    LightningModule's on_after_batch_transfer runs). Guards the DataModule/IterableDataset seam."""
    mega = _megasynth_cfg()
    dm = DataModule(_mixed({"a": mega, "b": mega}, {"a": 1.0, "b": 1.0}, "a"),
                    _data_loader_cfg(), StepTracker())
    loader = dm.train_dataloader()
    assert isinstance(loader.dataset, DatasetMixed)
    batch = next(iter(loader))
    ctx = BatchedViews.from_dict(batch["context"])
    tgt = BatchedViews.from_dict(batch["target"])
    assert ctx.image.ndim == 5 and ctx.image.shape[0] == 1  # [B, V, C, H, W]
    assert ctx.extrinsics.shape[-2:] == (4, 4)
    assert tgt.image.ndim == 5


# --------------------------------------------------------------------------- per-source init: tagging
@needs_megasynth
def test_mixer_tags_source_train():
    """Train items are tagged with the drawn source key (so the SceneTrainer can dispatch)."""
    mega = _megasynth_cfg()
    dm = DatasetMixed(_mixed({"a": mega, "b": mega}, {"a": 1.0, "b": 1.0}, "a"), "train", StepTracker())
    it = iter(dm)
    assert all(next(it)["source"] in {"a", "b"} for _ in range(10))


@needs_megasynth
def test_mixer_tags_source_val_is_eval_source():
    mega = _megasynth_cfg()
    dm = DatasetMixed(_mixed({"a": mega, "b": mega}, {"a": 1.0, "b": 1.0}, "b"), "val", StepTracker())
    assert next(iter(dm))["source"] == "b"


# --------------------------------------------------------------------------- per-source init: config
def test_mixed_dense_config_resolves_per_source_inits():
    """@package per-source init composition + dacite dict[str, InitializerCfg] resolution."""
    st = load_typed_root_config(_compose(["+experiment=train_mixed_dense"])).scene_trainer
    assert isinstance(st.scene_initializer, InitializerColmapCfg)  # default (dl3dv + eval)
    assert isinstance(st.scene_initializers["mega"], InitializerDepthCfg)  # override
    assert st.scene_initializers["mega"].init_longer_side == 80


def test_single_dataset_has_no_source_inits():
    """Additive: a single-dataset config leaves scene_initializers None (byte-identical path)."""
    st = load_typed_root_config(_compose(["+experiment=test_re10k", "output_dir=/tmp/x"])).scene_trainer
    assert st.scene_initializers is None


def test_compat_assert_rejects_mismatched_gaussian_params():
    """A per-source init with a different sh_degree can't share the one optimizer -> fail at load."""
    cfg = _compose(["+experiment=train_mixed_dense", "scene_trainer.scene_initializers.mega.sh_degree=2"])
    with pytest.raises(AssertionError, match="sh_degree/init_gaussian_multiple"):
        load_typed_root_config(cfg)


def test_key_subset_validation_rejects_unknown_source():
    """A per-source init keyed to a non-existent source is a typo -> fail fast at load."""
    cfg = _compose(["+experiment=train_mixed_dense"])
    OmegaConf.set_struct(cfg, False)
    cfg.scene_trainer.scene_initializers.bogus = deepcopy(cfg.scene_trainer.scene_initializers.mega)
    with pytest.raises(AssertionError, match="not dataset.sources"):
        load_typed_root_config(cfg)


def test_get_eval_cfg_drops_source_inits():
    """Eval collapses to the single eval source, so per-source overrides are dropped (else the
    mixed-only check would reject the single-source eval cfg)."""
    eval_cfg = get_eval_cfg(_compose(["+experiment=train_mixed_dense"]))
    assert isinstance(eval_cfg.dataset, DatasetDL3DVCfg)
    assert eval_cfg.scene_trainer.scene_initializers is None


# --------------------------------------------------------------------------- per-source init: dispatch
def test_initializer_for_dispatch():
    """SceneTrainer._initializer_for routes by source: override -> its init; unlisted / untagged /
    no-overrides -> the default. Uses a real nn.ModuleDict (as SceneTrainer builds), so the dispatch is
    exercised against ModuleDict semantics -- nn.ModuleDict has no .get(), unlike a plain dict."""
    import torch.nn as nn
    stub = SceneTrainer.__new__(SceneTrainer)
    nn.Module.__init__(stub)  # so nn.Module/ModuleDict attribute assignment works on the bare instance
    default, depth = nn.Identity(), nn.Identity()
    stub.initializer = default
    stub.source_initializers = nn.ModuleDict()  # no overrides -> always default
    assert stub._initializer_for({"source": ["mega"]}) is default
    stub.source_initializers = nn.ModuleDict({"mega": depth})
    assert stub._initializer_for({"source": ["mega"]}) is depth       # override hit
    assert stub._initializer_for({"source": ["dl3dv"]}) is default    # unlisted -> fallback
    assert stub._initializer_for({}) is default                       # eval path: no source tag


# --------------------------------------------------------------------------- per-source (modality) loss
def test_render_depth_loss_is_modality_gated():
    """LossRenderDepth (one global weight) contributes only when GT depth is present, and is exactly
    0 for a source without the depth modality -- the gate the mixed-dense run relies on. Depends only
    on modality availability (gt_depth is None), not on any source key."""
    from learn2splat.loss import LossRenderDepthCfgWrapper
    from learn2splat.loss.loss_render_depth import LossRenderDepth, LossRenderDepthCfg
    from learn2splat.model.decoder.decoder import DecoderOutput

    loss = LossRenderDepth(LossRenderDepthCfgWrapper(render_depth=LossRenderDepthCfg(weight=0.05)))
    pred = DecoderOutput(color=torch.rand(1, 1, 3, 4, 4), depth=torch.full((1, 1, 4, 4), 2.0))
    near, far = torch.full((1, 1, 1, 1), 0.1), torch.full((1, 1, 1, 1), 100.0)
    assert loss(pred, None, 0, gt_depth=None, near=near, far=far).item() == 0.0  # no modality -> 0
    gt = torch.full((1, 1, 4, 4), 1.0)  # present + mismatched render -> positive
    assert loss(pred, None, 0, gt_depth=gt, depth_mask=None, near=near, far=far).item() > 0.0


def test_curr_gt_depth_none_without_modality():
    """The extraction feeding the depth loss returns all-None when a source has no depth key (so the
    loss sees gt_depth=None) and the tensors when it does -- purely `views.get("depth")`-driven."""
    from learn2splat.meta_trainer.meta_trainer import MetaTrainer
    assert MetaTrainer._curr_gt_depth({}, None) == (None, None, None, None)
    views = {"depth": torch.rand(1, 2, 4, 4), "near": torch.full((1, 2), 0.1),
             "far": torch.full((1, 2), 100.0)}
    gt, mask, near, far = MetaTrainer._curr_gt_depth(views, None)
    assert gt is not None and mask is None and near.shape == (1, 2, 1, 1)


def test_mixed_dense_enables_render_depth_loss():
    from learn2splat.loss import LossRenderDepthCfgWrapper
    root = load_typed_root_config(_compose(["+experiment=train_mixed_dense"]))
    assert any(isinstance(w, LossRenderDepthCfgWrapper) for w in root.loss)
