from typing import Any, Dict

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from lightning import pytorch as pl
from src.build.from_config import instantiate_from_config
from src.util import rank_zero_log_only
import logging
logger = logging.getLogger(__name__)

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, dataset_name: str, train_batch_size: int, val_batch_size: int = 1, test_batch_size: int = 1,
                 train: Dict[str, Any] = None, validation: Dict[str, Any] = None, test: Dict[str, Any] = None,
                 wrap: bool = False, num_workers: int = 4):
        super().__init__()
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.save_hyperparameters()

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            rank_zero_log_only(logger, f"Dataset configuration: {OmegaConf.to_yaml(data_cfg)}")
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.train_batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers, shuffle=False)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.test_batch_size,
                          num_workers=self.num_workers, shuffle=False)
