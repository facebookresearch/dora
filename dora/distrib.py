# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import logging
import os

import submitit
import torch

from .xp import get_xp

logger = logging.getLogger(__name__)


DistribSpec = namedtuple(
    "DistribSpec", "rank world_size local_rank node_rank num_nodes source")


def get_distrib_spec():
    """Return information on the distributed setup, i.e. world size, rank etc.
    This can be used even before distributed training is initialized, which is useful for
    PytorchLightning for instance.
    """
    try:
        env = submitit.JobEnvironment()
    except RuntimeError:
        if 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = rank
            node_rank = 0
            num_nodes = 1
            source = "env"
        else:
            rank = 0
            world_size = 1
            local_rank = 0
            node_rank = 0
            num_nodes = 1
            source = "empty"
    else:
        rank = env.global_rank
        world_size = env.num_tasks
        local_rank = env.local_rank
        node_rank = env.node
        num_nodes = env.num_nodes
        source = "submitit"
    return DistribSpec(rank, world_size, local_rank, node_rank, num_nodes, source)


def init(backend='nccl'):
    """
    Initialize DDP.
    """
    spec = get_distrib_spec()
    if spec.world_size == 1:
        logger.info("world_size is 1, skipping init.")
        return
    xp = get_xp()
    torch.cuda.set_device(spec.local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        init_method='file://' + os.path.abspath(xp.rendezvous_file),
        world_size=spec.world_size,
        rank=spec.rank)
    logger.info(
        "Distributed init: %d/%d (local %d) from %s",
        spec.rank, spec.world_size, spec.local_rank, spec.source)


def is_master():
    return rank() == 0


def rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1
