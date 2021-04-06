# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from .main import get_xp


def init(backend='nccl'):
    """
    Initialize DDP.
    """
    if 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        return
    if world_size == 1:
        return
    xp = get_xp()
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        init_method='file://' + os.path.abspath(xp.rendezvous_file),
        world_size=world_size,
        rank=rank)


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
