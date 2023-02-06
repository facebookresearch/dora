# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import logging
import os
import random
import subprocess as sp

import submitit
import torch

from .xp import get_xp

logger = logging.getLogger(__name__)


DistribSpec = namedtuple(
    "DistribSpec", "rank world_size local_rank node_rank num_nodes source")


def set_distrib_env():
    """Calling this function will set the distributed environement
    including the master addr, master port etc. You shouldn't call
    this if you call `dora.distrib.init`, but it can be useful if you need to let
    some other framework handle the distributed initialization.
    """
    spec = get_distrib_spec()
    if spec.world_size == 1:
        return
    if 'MASTER_ADDR' not in os.environ:
        assert 'SLURM_JOB_NODELIST' in os.environ, "case not handled"
        nodelist = os.environ['SLURM_JOB_NODELIST']
        nodes = sp.run('scontrol show hostnames'.split() + [nodelist],
                       capture_output=True, check=True).stdout.decode().split()
        master_node = nodes[0]
        os.environ['MASTER_ADDR'] = master_node
    if 'MASTER_PORT' not in os.environ:
        xp = get_xp()
        # Note that running twice the same XP on the same node will crash,
        # but that shouldn't really happen
        seed = xp.sig
        # If we are in a Slurm job, let us use the Slurm job id.
        try:
            env = submitit.JobEnvironment()
        except RuntimeError:
            pass
        else:
            seed += env.job_id
        rng = random.Random(seed)
        master_port = rng.randint(20000, 60000)
        os.environ['MASTER_PORT'] = str(master_port)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(spec.world_size)
        os.environ['RANK'] = str(spec.rank)
        os.environ['LOCAL_RANK'] = str(spec.local_rank)


def get_distrib_spec():
    """Return information on the distributed setup, i.e. world size, rank etc.
    This can be used even before distributed training is initialized, which is useful for
    PytorchLightning for instance.
    """
    if 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            local_rank = rank
        node_rank = 0
        num_nodes = 1
        source = "env"
    else:
        try:
            env = submitit.JobEnvironment()
        except RuntimeError:
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
    if torch.distributed.is_initialized():
        return
    spec = get_distrib_spec()
    if spec.world_size == 1:
        logger.info("world_size is 1, skipping init.")
        return
    xp = get_xp()
    if torch.cuda.is_available():
        torch.cuda.set_device(spec.local_rank)
    else:
        assert backend != 'nccl'

    if xp.dora.use_rendezvous:
        init_method = 'file://' + os.path.abspath(xp.rendezvous_file)
    else:
        set_distrib_env()
        init_method = 'env://'
    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=spec.world_size,
        rank=spec.rank)
    logger.info(
        "Distributed init: %d/%d (local %d) from %s",
        spec.rank, spec.world_size, spec.local_rank, spec.source)
    if xp.dora.use_rendezvous:
        torch.distributed.barrier()
        if rank() == 0:
            # Delete rendez vous file early, let's hope this doesn't bug too much.
            xp.rendezvous_file.unlink()


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
