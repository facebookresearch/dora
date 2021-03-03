"""
Basic configuration for dora is here.
"""
from argparse import Namespace
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
import typing as tp


def update_from_args(data: tp.Any, args: Namespace):
    """Update the given dataclass from the argument parser args.
    """
    for key in data.__dict__:
        if hasattr(args, key):
            setattr(data, key, getattr(args, key))


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
        cpus_per_task (int): number of cpus per task.
        partition (str): partition name
        comment (str): comment for the job.
        setup (List[str]): list of shell commands to execute
            before the actual command. Use it for `module load`.
        max_num_timeout (int): maximum number of requeue.

    ..warning:: this assumes one task per GPU. Support for
        multiple GPUs per task might be added in the future.
        Same for no GPU tasks.
    """
    gpus: int = 1
    mem_per_gpu: float = 10
    time: int = 1200
    cpus_per_task: int = 10
    partition: str = "learnfair"
    comment: tp.Optional[str] = None
    setup: tp.List[str] = field(default_factory=list)

    max_num_timeout: int = 20


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
    grids: str = "grids"
    submitit_folder: str = "submitit"


@dataclass
class DoraConfig:
    """
    Main Dora configuration. The main parameters to change are the following.

    Args:
        name (str): basename for the experiments. This will be preprended
            to the name of all experiments.
        dir (Path): path where Dora will save all useful informations, logs.
            This is also where you should store your checkpoints (see `dora.xp.XP`).
        exclude (List[str]): list of patterns of argument names to ignore
            when computing the XP signature and doing deduplication.
            For instance 'num_workers', etc.
    """

    name: str = "default"  # default basename for experiments
    dir: Path = Path("./outputs")  # where everything will be stored
    exclude: tp.List[str] = field(default_factory=list)

    history: str = "history.json"  # where metrics will be stored
    runs: str = "runs"

    shep: ShepConfig = field(default_factory=ShepConfig)
    rendezvous_file: str = "rendezvous.txt"

    def is_excluded(self, arg_name: str) -> bool:
        """Return True if the given argument name should be excluded from
        the signature."""
        for pattern in self.exclude:
            if fnmatch(arg_name, pattern):
                return True
        return False

