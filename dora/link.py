import json
import logging

from . import distrib
from . import utils

logger = logging.getLogger(__name__)


class Link:
    """
    Connection with Dora for your trainer.
    This is minimalistic and won't do much.
    """
    def __init__(self, run):
        """
        Initialize the Link with Dora.
        """
        self.run = run
        self.history = []
        self.history_file = run.folder / run.dora.history

    def setup(self, load_history: bool = True):
        if load_history and self.history_file.exists():
            history = utils.try_load(self.history_file, load=json.load, mode='r')
            if history is not None:
                self.history = history

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
