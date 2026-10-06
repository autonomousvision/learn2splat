"""Mixed-dataset training: a probability-weighted multiplexer over independent dataset readers.

`DatasetMixed` composes several existing readers (each built via `get_dataset`, so it keeps its own
format, view-sampler and shims) and, during training, interleaves their per-scene examples by a
per-source weight. Val/test delegate to a single configured `eval_source` (mirrors the DL3DV
`no_mix_test_set` convention), so eval numbers match running that source alone.

Only active for `dataset=mixed`; the single-dataset path is unchanged. Config: `config/dataset/
mixed.yaml` + an experiment that composes the sources via Hydra `@package` (see
`config/experiment/train_mixed.yaml`).
"""
from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import IterableDataset, get_worker_info

from ..misc.step_tracker import StepTracker
from .data_types import Stage
from .dataset_colmap import DatasetColmapCfg
from .dataset_dl3dv import DatasetDL3DVCfg
from .dataset_megasynth import DatasetMegasynthCfg
from .dataset_re10k import DatasetRE10kCfg
from .dataset_scannet import DatasetScannetCfg

# The leaf dataset cfgs a mixed run can compose (excludes MixedDatasetCfg -> non-recursive union).
SingleDatasetCfg = (
    DatasetRE10kCfg | DatasetDL3DVCfg | DatasetMegasynthCfg | DatasetColmapCfg | DatasetScannetCfg
)


@dataclass
class MixedDatasetCfg:
    name: Literal["mixed"]
    sources: dict[str, SingleDatasetCfg]  # keyed sources; each a complete leaf dataset cfg
    weights: dict[str, float]             # keyed sampling weights, parallel to `sources`
    eval_source: str                      # key into `sources`; drives val/test (single eval set)
    background_color: list[float]         # decoder background; the only global field. Each source's
                                          # own background_color is ignored (the decoder uses this).


class DatasetMixed(IterableDataset):
    """Weighted multiplexer over independent readers (train); a single eval source (val/test).

    Every per-source setting (image_shape, view_sampler, near/far, augment, ...) is honoured because
    each child is built from its complete source cfg; the mixer only adds the weighted interleave.
    """

    def __init__(self, cfg: MixedDatasetCfg, stage: Stage, step_tracker: StepTracker | None) -> None:
        super().__init__()
        self.cfg = cfg
        from . import get_dataset  # lazy: dataset_mixed is imported by the package __init__

        # `self.probs is None` is the mode flag used everywhere below: None -> delegate to the single
        # eval source (val/test); set -> weighted multiplex over all sources (train).
        if stage in ("val", "test"):
            # Evaluate on one source only -> eval numbers identical to running it alone.
            src = cfg.sources[cfg.eval_source]
            self.children = [get_dataset(src, stage, step_tracker)]
            self.probs = None
        else:
            self.keys = list(cfg.sources.keys())  # draw index -> source key, for tagging items
            self.children = [get_dataset(cfg.sources[k], stage, step_tracker) for k in self.keys]
            w = torch.tensor([float(cfg.weights[k]) for k in self.keys], dtype=torch.float64)
            assert (w >= 0).all() and w.sum() > 0, "mixed dataset weights must be >=0 with a positive sum"
            self.probs = w / w.sum()

    def __iter__(self):
        # Tag each item with its source key so the SceneTrainer can dispatch a per-source initializer
        # (see scene_trainer.scene_initializers). Single datasets never set this key.
        if self.probs is None:  # val/test: delegate to the single eval source
            for item in self.children[0]:
                item["source"] = self.cfg.eval_source
                yield item
            return
        # Train: draw a source per item by weight; reopen an exhausted child so the stream is
        # effectively infinite (the trainer stops via max_steps). Seed the draw off the worker so
        # different workers pick different sequences (each child still shards/shuffles itself).
        gen = torch.Generator()
        info = get_worker_info()
        # Advance the seed on each __iter__ call so the source sequence doesn't restart
        # from the same draw every epoch (persistent workers keep info.seed constant across
        # epochs; without this, the first batch of every epoch draws the same source).
        self._iter_count = getattr(self, '_iter_count', -1) + 1
        base_seed = int(info.seed) if info is not None else torch.initial_seed()
        gen.manual_seed(base_seed + self._iter_count)
        iters = [iter(c) for c in self.children]
        while True:
            k = int(torch.multinomial(self.probs, 1, generator=gen).item())
            try:
                item = next(iters[k])
            except StopIteration:  # child exhausted -> reopen; a fresh empty child is a real error
                iters[k] = iter(self.children[k])
                item = next(iters[k])
            item["source"] = self.keys[k]
            yield item

    def __len__(self) -> int:
        # Raises TypeError if a source has no __len__ (e.g. megasynth) -> callers treat the mixer as
        # a streaming IterableDataset, same as that source alone. dl3dv/re10k sources give a length.
        if self.probs is None:  # val/test: single eval source
            return len(self.children[0])
        return sum(len(c) for c in self.children)
