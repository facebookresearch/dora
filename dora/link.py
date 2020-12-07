import json
import logging
from pathlib import Path
import random

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch

from . import distrib
from . import utils

logger = logging.getLogger(__name__)


class Link:
    """
    Connection with Dora for your trainer.
    This is minimalistic and won't do much.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize the Link with Dora.
        if `load` is True, automatically loads the history from any
        previous `history` file.
        """
        self.cfg = cfg
        self.history = []
        self.history_file = self._path(cfg.dora.history)

    def setup(self, load_history: bool = True, init_seed: bool = True, init_distrib: bool = True):
        if load_history and self.history_file.exists():
            history = utils.try_load(self.history_file, load=json.load, mode='r')
            if history is not None:
                self.history = history
        if init_seed:
            random.seed(self.cfg.dora.seed)
            torch.manual_seed(self.cfg.dora.seed)

        if init_distrib:
            distrib.init(**self.cfg.dora.ddp)

    def _path(self, path):
        """
        Get the path relative to Hydra execution folder. This is needed if we are
        creating the backbone from outside Hydra.main, for instance a notebook.
        This allows to recreate the object perfectly for interactive use.
        """
        if HydraConfig.initialized():
            # We are running from Hydra.main
            return Path(path)
        else:
            # Get the directory from cfg
            return Path(self.cfg.hydra.run.dir) / path

    def _commit(self):
        if not distrib.is_master():
            return
        with utils.write_and_rename(self.history_file, "w") as tmp:
            json.dump(self.history, tmp)

    def update_history(self, history):
        self.history = utils.jsonable(history)
        self._commit()

    def push_metrics(self, metrics):
        metrics = utils.jsonable(metrics)
        self.history.append(metrics)
        self._commit()
