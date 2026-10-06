import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from . import DatasetCfg, get_dataset
from .data_types import DataShim, Stage
from .validation_wrapper import ValidationWrapper
from ..misc.step_tracker import StepTracker


def get_data_shim(initializer: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(initializer, "get_data_shim"):
        shims.append(initializer.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2 ** 32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
            self,
            dataset_cfg: DatasetCfg,
            data_loader_cfg: DataLoaderCfg,
            step_tracker: StepTracker | None = None,
            dataset_shim: DatasetShim = lambda dataset, _: dataset,
            global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        loader_cfg = self.data_loader_cfg.train

        dataset = get_dataset(
            self.dataset_cfg,
            "train",
            self.step_tracker,
        )
        dataset = self.dataset_shim(dataset, "train")

        return DataLoader(
            dataset,
            loader_cfg.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=loader_cfg.num_workers,
            generator=self.get_generator(loader_cfg),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(loader_cfg),
        )

    def val_dataloader(self):
        loader_cfg = self.data_loader_cfg.val

        dataset = get_dataset(
            self.dataset_cfg,
            "val",
            self.step_tracker,
        )
        dataset = self.dataset_shim(dataset, "val")

        return DataLoader(
            ValidationWrapper(dataset, 1),
            loader_cfg.batch_size,
            num_workers=loader_cfg.num_workers,
            generator=self.get_generator(loader_cfg),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(loader_cfg),
        )

    def test_dataloader(self, dataset_cfg=None):
        loader_cfg = self.data_loader_cfg.test

        dataset = get_dataset(
            self.dataset_cfg if dataset_cfg is None else dataset_cfg,
            "test",
            self.step_tracker,
        )
        dataset = self.dataset_shim(dataset, "test")

        return DataLoader(
            dataset,
            loader_cfg.batch_size,
            num_workers=loader_cfg.num_workers,
            generator=self.get_generator(loader_cfg),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(loader_cfg),
            shuffle=False,
        )
