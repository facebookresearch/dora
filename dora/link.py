import json
import logging

from . import utils

logger = logging.getLogger(__name__)


class Link:
    """
    Connection with Dora for your trainer.
    This is minimalistic and won't do much.
    """
    def __init__(self, xp):
        """
        Initialize the Link with Dora.
        """
        self.xp = xp
        self.history = []
        self.history_file = xp.folder / xp.dora.history

    def load(self):
        if self.history_file.exists():
            history = utils.try_load(self.history_file, load=json.load, mode='r')
            if history is not None:
                self.history = history

    def _commit(self):
        from . import distrib
        if not distrib.is_master():
            return
        with utils.write_and_rename(self.history_file, "w") as tmp:
            json.dump(self.history, tmp, indent=2)

    def update_history(self, history):
        history = utils.jsonable(history)
        if not isinstance(history, list):
            raise ValueError(f"history must be a list, but got {type(history)}")
        self.history = history
        self._commit()

    def push_metrics(self, metrics):
        metrics = utils.jsonable(metrics)
        self.history.append(metrics)
        self._commit()
