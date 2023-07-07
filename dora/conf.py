# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Basic configuration for Dora is here.
"""
from argparse import Namespace
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
import typing as tp

from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf


def update_from_args(data: tp.Any, args: Namespace):
    """Update the given dataclass from the argument parser args.
    """
    for key in data.__dict__:
        assert isinstance(key, str)
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                setattr(data, key, value)


def update_from_hydra(data: tp.Any, cfg: DictConfig):
    """Update the given dataclass from the hydra config.
    """

    dct = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(dct, dict)
    for key, value in dct.items():
        assert isinstance(key, str)
        if hasattr(data, key):
            setattr(data, key, value)
        else:
            raise AttributeError(f"Object of type {data.__class__} "
                                 f"does not have an attribute {key}")


@dataclass
class SlurmConfig:
    """
    Configuration when scheduling a job.
    This differs slightly from Slurm/Submitit params because this will
    automatically scale some values based on the number of GPUs.

    Args:
        gpus (int): number of total GPUs to schedule. Number of nodes
            and tasks per nodes will be automatically inferred.
        mem_per_gpu (float): amount of memory in GB to schedule
            per gpus.
        time (int): maximum duration for the job in minutes.
        cpus_per_gpu (int): number of cpus per gpu, this will set
            the `cpus_per_task` automatically, based on the
            number of gpus and `one_task_per_node`, unless `cpus_per_task`
            is explicitely provided.
        cpus_per_task (int or None): number of cpus per task.
        partition (str): partition name
        comment (str): comment for the job.
        setup (List[str]): list of shell commands to execute
            before the actual command. Use it for `module load`.
        max_num_timeout (int): maximum number of requeue.
        one_task_per_node (bool): if True, schedules a single task
            per node, otherwise, will schedule one task per gpu (default is False).
        array_parallelism (int): when using job arrays, how many tasks can run
            in parallel.
        qos: (str or None): qos param for slurm.
        account: (str or None): account param for slurm.

    ..warning:: this assumes one task per GPU.
        Set `one_task_per_node` if you do not want that.
        Tasks without any gpus are not really supported at the moment.
    """
    gpus: int = 1
    mem_per_gpu: float = 40
    time: int = 1200
    cpus_per_gpu: int = 10
    cpus_per_task: tp.Optional[int] = None
    partition: str = "learnlab"
    comment: tp.Optional[str] = None
    setup: tp.List[str] = field(default_factory=list)
    max_num_timeout: int = 20
    constraint: str = ""
    one_task_per_node: bool = False
    array_parallelism: int = 256
    exclude: tp.Optional[str] = None
    qos: tp.Optional[str] = None
    account: tp.Optional[str] = None


@dataclass
class SubmitRules:
    """
    Submit rules describe in which case Shepherd will schedule new jobs.

    Args:
        retry (bool): if true, all failed or canceled jobs will be rescheduled.
        update_pending (bool): if true, all pending jobs whose Slurm parameters
            have changed will be replaced.
        update (bool): if true, all pending or running jobs whose Slurm parameters
            have changed will be replaced.
        replace (bool): if true, all running jobs will be replaced by new jobs.
        replace_done (bool): if true, all done jobs will be relaunched.
    """

    retry: bool = False
    update: bool = False
    replace: bool = False
    replace_done: bool = False


@dataclass
class ShepConfig:
    """
    Configuration for Shepherd. Mostly naming conventions for folders and files.
    There should be little reasons to change that.
    """
    job_file: str = "job.pkl"
    by_id: str = "by_id"
    orphans: str = "orphans"
    submitit_folder: str = "submitit"
    latest_submitit: str = "latest"
    arrays: str = "arrays"


@dataclass
class DoraConfig:
    """
    Main Dora configuration. The main parameters to change are the following.

    Args:
        dir (Path or str): path where Dora will save all useful informations, logs.
            This is also where you should store your checkpoints (see `dora.xp.XP`).
        exclude (List[str]): list of patterns of argument names to ignore
            when computing the XP signature and doing deduplication.
            For instance 'num_workers', etc.
        git_save (bool): when True, experiments can only be scheduled from a clean repo.
            A shallow clone of the repo will be made and execution will happen from there.
            This does not impact `dora run` unless you pass the `--git_save` flag.
        shared (Path or None): if provided, the path to a central repository of XPs.
            For the moment, this only supports sharing hyper-params, logs etc. will stay
            in the per user folder.
        grid_package (str or None): if provided, package to look for grids. Default
            to the package with the `train.py` module followed by `.grids`.
    """
    dir: Path = Path("./outputs")  # where everything will be stored
    exclude: tp.List[str] = field(default_factory=list)
    git_save: bool = False
    shared: tp.Optional[Path] = None  # Optional path for shared XPs.
    grid_package: tp.Optional[str] = None

    # Those are internal config values and are unlikely to be changed
    history: str = "history.json"  # where metrics will be stored
    xps: str = "xps"  # subfolder to store xps

    shep: ShepConfig = field(default_factory=ShepConfig)
    rendezvous_file: str = "rendezvous.txt"
    use_rendezvous: bool = False
    # Filenames used in various places, you shouldn't edit that
    _grids: str = "grids"
    _codes: str = "codes"

    def is_excluded(self, arg_name: str) -> bool:
        """Return True if the given argument name should be excluded from
        the signature."""
        for pattern in self.exclude:
            if fnmatch(arg_name, pattern):
                return True
        return False

    def __setattr__(self, name, value):
        if name in ['dir', 'shared']:
            from .git_save import to_absolute_path
            if value is not None:
                value = Path(to_absolute_path(value))
        super().__setattr__(name, value)
