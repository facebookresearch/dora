# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path
import typing as tp

from retrying import retry
from . import utils

logger = logging.getLogger(__name__)


class Link:
    """
    Connection with Dora for your trainer.
    This is minimalistic and won't do much.

    This can also be used to simulate a fake link by passing `None`
    as the history file.
    """
    def __init__(self, history_file: tp.Optional[Path] = None):
        """
        Initialize the Link with Dora.
        """
        self.history: tp.List[dict] = []
        self.history_file = history_file

    # Retry operation as history file might be stale for  update by running XP
    @retry(stop_max_attempt_number=10)
    def load(self):
        if self.history_file is None:
            return
        if self.history_file.exists():
            history = utils.try_load(self.history_file, load=json.load, mode='r')
            if history is not None:
                self.history = history

    def _commit(self):
        if self.history_file is None:
            return

        from . import distrib
        if not distrib.is_master():
            return
        with utils.write_and_rename(self.history_file, "w") as tmp:
            json.dump(self.history, tmp, indent=2)

    def update_history(self, history: tp.List[dict]):
        history = utils.jsonable(history)
        if not isinstance(history, list):
            raise ValueError(f"history must be a list, but got {type(history)}")
        self.history[:] = history
        self._commit()

    def push_metrics(self, metrics: dict):
        metrics = utils.jsonable(metrics)
        self.history.append(metrics)
        self._commit()
