# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
"""
Start multiple process locally for DDP.
"""

import logging
from pathlib import Path
import os
import subprocess as sp
import sys

from hydra import utils

logger = logging.getLogger(__name__)


class ChildrenManager:
    def __init__(self):
        self.children = []
        self.failed = False

    def add(self, child):
        child.rank = len(self.children)
        self.children.append(child)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            logger.error("An exception happened while starting workers %r", exc_value)
            self.failed = True
        try:
            while self.children and not self.failed:
                for child in list(self.children):
                    try:
                        exitcode = child.wait(0.1)
                    except sp.TimeoutExpired:
                        continue
                    else:
                        self.children.remove(child)
                        if exitcode:
                            logger.error(f"Worker {child.rank} died, killing all workers")
                            self.failed = True
        except KeyboardInterrupt:
            logger.error("Received keyboard interrupt, trying to kill all workers.")
            self.failed = True
        for child in self.children:
            child.terminate()
        if not self.failed:
            logger.info("All workers completed successfully")


def start_ddp_workers(package, main, overrides):
    import torch as th

    cfg = main.get_config(overrides)
    world_size = th.cuda.device_count()
    if not world_size:
        logger.error(
            "DDP is only available on GPU. Make sure GPUs are properly configured with cuda.")
        sys.exit(1)

    # If rendezvous file already exist, this can deadlock.
    rdv = Path(cfg.dora.ddp.rendezvous)
    if rdv.exists():
        rdv.unlink()
    print(f"Starting {world_size} worker processes for DDP.")
    with ChildrenManager() as manager:
        for rank in range(world_size):
            kwargs = {}
            env = dict(os.environ)
            env['DORA_RANK'] = str(rank)
            env['DORA_WORLD_SIZE'] = str(world_size)
            env['DORA_CHILD'] = '1'
            argv = ["-m", "dora", "-P", package]
            argv += overrides
            if rank > 0:
                kwargs['stdin'] = sp.DEVNULL
                kwargs['stdout'] = sp.DEVNULL
                kwargs['stderr'] = sp.DEVNULL
            manager.add(
                sp.Popen([sys.executable] + argv, cwd=utils.get_original_cwd(), env=env, **kwargs))
    sys.exit(int(manager.failed))
