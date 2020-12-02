import json
import logging
from pathlib import Path

from julius.utils import Chrono
from hydra.core.hydra_config import HydraConfig
import torch as th

from . import distrib
from . import utils

logger = logging.getLogger(__name__)


class Backbone:
    def __init__(self, cfg, model, optimizer):
        distrib.init(**cfg.dora.ddp)
        self.history = []
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.best_state = None

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

    def _get_state(self):
        """
        Return value to be serialized in the checkpoint.
        """
        return {
            "history": self.history,
            "model": utils.serialize_model(self.model),
            "cfg": self.cfg,
            "optimizer": self.optimizer.state_dict(),
            "best_state": self.best_state,
        }

    def _get_best_state(self):
        """
        Return value to be serialized in the best checkpoint.
        """
        assert self.best_state is not None
        with utils.swap_state(self.model, self.best_state):
            return {
                "history": self.history,
                "model": utils.serialize_model(self.model),
                "cfg": self.cfg,
            }

    def _set_state(self, state):
        """
        Given a state dict, reassign everything to the attribute of this instance.
        """
        self.history = state.get("history", [])
        utils.deserialize_model(state["model"], self.model)

        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])

        if "best_state" in state:
            self.best_state = state["best_state"]

    def resume(self):
        """
        Find the right place to load the state from.
        If a checkpoint exist, use it, otherwise if dora.continue_from is set,
        use it instead.
        Restore the state.
        """
        cfg = self.cfg
        if cfg.dora.restart:
            logger.debug("dora.restart is set, skipping checkpoint loading")
            return

        checkpoint_path = self._path(cfg.dora.checkpoint)
        state = utils.try_load(checkpoint_path)
        if state is not None:
            logger.debug("Resumed from checkpoint %s", checkpoint_path)
            self._set_state(state)
            return

        if cfg.dora.continue_from:
            # Are we passed something to resume from?
            # If it contains /, its a path, otherwise it's an experiment signature.
            if "/" in cfg.dora.continue_from:
                path = self._path(cfg.dora.continue_from)
            else:
                # We know all the experiments live in the same folder
                path = self._path("../" + cfg.dora.continue_from)
                if cfg.dora.continue_best:
                    path = path / cfg.dora.best
                else:
                    path = path / cfg.dora.checkpoint
            state = utils.try_load(path)
            if state is None:
                logger.error(
                    "Tried to continue from %s but could not load file %s.",
                    cfg.dora.continue_from, path)
            state.pop('history', None)  # Removing history.
            logger.debug("Continuing from %s", path)
            self._set_state(state)
            return

    def _save_checkpoint(self):
        """
        Actually save the checkpoint to disk.
        """
        cfg = self.cfg

        # Saving history as json for easier aggregation.
        with utils.write_and_rename(cfg.dora.history) as f:
            json.dump(self.history, f)

        if cfg.dora.checkpoint:
            # Storing full checkpoint as torch file.
            with utils.write_and_rename(cfg.dora.checkpoint) as f:
                th.save(self._get_state(), f)

        if cfg.dora.best:
            # Storing best state without the optimizer state etc.
            with utils.write_and_rename(cfg.dora.checkpoint) as f:
                th.save(self._get_best_state(), f)

    def train_epoch(self, epoch: int):
        """
        Training part of the epoch. Should return a dict of metrics.
        Tensors will be converted to lists automatically.
        """
        return {}

    def valid_epoch(self, epoch: int):
        """
        Validation part of the epoch.
        """
        return {}

    def test_epoch(self, epoch: int):
        """
        Test part of the epoch.
        """
        return {}

    def log_epoch(self, epoch, metrics):
        """
        Print a summary of the epoch to the log. This is in a separate function as
        this needs to be replayed when resuming from a previous checkpoint.
        """
        logger.info(f"End of Epoch {epoch}")

    def _update_best(self, metrics):
        cfg = self.cfg
        best_loss = None
        if cfg.dora.valid_key:
            loss = utils.get_metric(metrics, cfg.dora.valid_key)
            best_loss = float('inf')
            if self.history:
                best_loss = min(utils.pull_metric(self.history, cfg.dora.valid_key))
            if loss < best_loss:
                logger.info(f"New best loss {loss}, previous wav {best_loss}")
                best_loss = loss
                is_best = True
            else:
                is_best = False
        else:
            is_best = True
        if is_best:
            self.best_state = utils.copy_state(self.model.state_dict())
        return is_best

    def run(self):
        self.resume()

        if self.history:
            logger.info("Replaying epochs from a previous run.")
            for epoch, metrics in enumerate(self.history):
                self.log_epoch(epoch, metrics)

        for epoch in range(len(self.history), self.cfg.epochs):
            metrics = {}
            # We first do training and valid.
            with Chrono() as chrono:
                metrics["train"] = self.train_epoch(epoch)
            metrics["train"]["duration"] = chrono.duration

            with Chrono() as chrono:
                metrics["valid"] = self.valid_epoch(epoch)
            metrics["valid"]["duration"] = chrono.duration

            self._update_best(metrics)

            with Chrono() as chrono:
                with utils.swap_state(self.model, self.best_state):
                    metrics["test"] = self.test_epoch(epoch)
            metrics["test"]["duration"] = chrono.duration

            self.log_epoch(epoch, metrics)
            self.history.append(utils.jsonable(metrics))
            self._save_checkpoint()
