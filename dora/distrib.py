# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel.distributed import DistributedDataParallel

logger = logging.getLogger(__name__)
rank = 0
world_size = 1


def init(args):
    """init.

    Initialize DDP using the given rendezvous file.
    """
    global rank, world_size
    if args.ddp:
        assert args.rank is not None and args.world_size is not None
        rank = args.rank
        world_size = args.world_size
    if world_size == 1:
        return
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend=args.ddp_backend,
        init_method='file://' + os.path.abspath(args.rendezvous_file),
        world_size=world_size,
        rank=rank)
    logger.debug("Distributed rendezvous went well, rank %d/%d", rank, world_size)


def average(metrics, count=1.):
    """average.

    Average all the relevant metrices across processes
    `metrics`should be a 1D float32 fector. Returns the average of `metrics`
    over all hosts. You can use `count` to control the weight of each worker.
    """
    if world_size == 1:
        return metrics
    tensor = torch.tensor(list(metrics) + [1], device='cuda', dtype=torch.float32)
    tensor *= count
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return (tensor[:-1] / tensor[-1]).cpu().numpy().tolist()


def sync_grad(params):
    """
    Simpler alternative to DistributedDataParallel, that doesn't rely
    on any black magic. For simple models it can also be as fast.
    Just call this on your model parameters after the call to backward!
    """
    if world_size == 1:
        return
    handles = []
    for p in params:
        if p.grad is not None:
            handle = torch.distributed.all_reduce(
                p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
            handles.append((p, handle))
    for p, handle in handles:
        handle.wait()
        p.grad.data /= world_size


def wrap(model):
    """wrap.

    Wrap a model with DDP if distributed training is enabled.
    """
    if world_size == 1:
        return model
    else:
        return DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device())


def barrier():
    if world_size > 1:
        torch.distributed.barrier()


def share(obj=None):
    if world_size == 1:
        return obj
    size = torch.empty(1, device='cuda', dtype=torch.long)
    if rank == 0:
        dump = pickle.dumps(obj)
        size[0] = len(dump)
    torch.distributed.broadcast(size, src=0)
    if rank == 0:
        buffer = torch.from_numpy(np.frombuffer(dump, dtype=np.uint8).copy()).cuda()
    else:
        buffer = torch.empty(size[0].item(), device='cuda', dtype=torch.uint8)
    torch.distributed.broadcast(buffer, src=0)
    if rank > 0:
        obj = pickle.loads(buffer.cpu().numpy().tobytes())
    logger.debug(f"Shared object of size {len(buffer)}")
    return obj


def loader(dataset, *args, shuffle=False, klass=DataLoader, **kwargs):
    """loader.

    Create a dataloader properly in case of distributed training.
    If a gradient is going to be computed you must set `shuffle=True`.

    :param dataset: the dataset to be parallelized
    :param args: relevant args for the loader
    :param shuffle: shuffle examples
    :param klass: loader class
    :param kwargs: relevant args
    """

    if world_size == 1:
        return klass(dataset, *args, shuffle=shuffle, **kwargs)

    if shuffle:
        # train means we will compute backward, we use DistributedSampler
        sampler = DistributedSampler(dataset)
        # We ignore shuffle, DistributedSampler already shuffles
        return klass(dataset, *args, **kwargs, sampler=sampler)
    else:
        # We make a manual shard, as DistributedSampler otherwise replicate some examples
        dataset = Subset(dataset, list(range(rank, len(dataset), world_size)))
        return klass(dataset, *args, **kwargs, shuffle=shuffle)
