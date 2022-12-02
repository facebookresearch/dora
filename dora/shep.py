# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Scheduling and job monitoring utilities.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
import logging
from pathlib import Path
import pickle
import os
import subprocess as sp
import sys
import typing as tp


from submitit import SlurmJob
import submitit

from . import git_save
from .conf import SlurmConfig, SubmitRules
from .distrib import get_distrib_spec
from .main import DecoratedMain
from .utils import try_load
from .xp import XP, _get_sig


logger = logging.getLogger(__name__)


class _SubmitItTarget:
    def __call__(self, main: DecoratedMain, argv: tp.Sequence[str]):
        self.xp = main.get_xp(argv)
        spec = get_distrib_spec()
        # We export the RANK as it can be used to customize logging early on
        # in the called program (e.g. using Hydra).
        os.environ['RANK'] = str(spec.rank)
        sys.argv[1:] = argv
        main()

    def checkpoint(self, *args, **kwargs):
        if get_distrib_spec().rank == 0:
            # cleanup rendezvous file on requeue, otherwise things will fail.
            if self.xp.rendezvous_file.exists():
                self.xp.rendezvous_file.unlink()
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


class Sheep:
    """
    A Sheep is a specific run for a given XP. Sheeps are managed
    by the Shepherd.
    """
    def __init__(self, xp: XP):
        self.xp = xp
        self.job: tp.Optional[submitit.SlurmJob] = None
        # Other jobs contain the list of other jobs in the array
        self._other_jobs: tp.Optional[tp.List[submitit.SlurmJob]] = None
        if self._job_file.exists():
            content = try_load(self._job_file)
            if isinstance(content, tuple):
                self.job, self._other_jobs = content
            else:
                self.job = content

    @property
    def _job_file(self) -> Path:
        return self.xp.folder / self.xp.dora.shep.job_file

    def state(self, mode="standard"):
        """Return the current state of the `Sheep`.
        """
        if self.job is None:
            return None
        state = self.job.watcher.get_state(self.job.job_id, mode)
        if state == 'UNKNOWN' and self._other_jobs:
            if any(job.state != 'UNKNOWN' for job in self._other_jobs):
                # When cancelling single entries in a job array,
                # sacct will just completely forget about it insted of marking
                # it as cancelled. So we use a specific 'MISSING' status to handle that.
                state = 'MISSING'
        if state.startswith('CANCELLED'):
            return 'CANCELLED'
        return state

    def is_done(self, mode="standard"):
        """Return True if the job is no longer running on the cluster.
        """
        if self.job is None:
            return True
        return self.job.watcher.is_done(self.job.job_id, mode)

    @property
    def log(self):
        """Return the path to the main log.
        """
        if self.job is not None:
            return self.xp.submitit / f"{self.job.job_id}_0_log.out"
        return None

    def __repr__(self):
        out = f"Sheep({self.xp.sig}, state={self.state()}, "
        if self.job is not None:
            out += f"sid={self.job.job_id}, "

        out += f"argv={self.xp.argv})"
        return out


def no_log(x: str):
    """No logging logging function, passed to `Shepherd`.
    """
    pass


@dataclass
class _JobArray:
    slurm_config: SlurmConfig
    sheeps: tp.List[Sheep] = field(default_factory=list)


class Shepherd:
    """
    Takes care of the little jobs.

    Args:
        main (DecoratedMain): main function decorated by Dora.
        log (callable): log function, if provided should take a single string
            argument.
    """
    def __init__(self, main: DecoratedMain, log: tp.Callable[[str], None] = no_log):
        self.main = main
        self._by_id.mkdir(exist_ok=True, parents=True)
        self._orphans.mkdir(exist_ok=True, parents=True)
        self._arrays.mkdir(exist_ok=True, parents=True)
        self.log = log

        self._in_job_array: bool = False
        self._existing_git_clone: tp.Optional[Path] = None
        self._to_cancel: tp.List[submitit.SlurmJob] = []
        self._to_submit: tp.List[_JobArray] = []

        self._check_orphans()

    def get_sheep_from_argv(self, argv: tp.Sequence[str]) -> Sheep:
        """
        Given a list of of arguments, return the matching `Sheep`,
        which will contain both information on the `dora.xp.XP`, and on
        the latest job associated with that XP.
        """
        assert not isinstance(argv, str)
        xp = self.main.get_xp(argv)
        return Sheep(xp)

    def get_sheep_from_sig(self, sig: str) -> tp.Optional[Sheep]:
        """
        Returns a `Sheep` given the XP signature, if any exists, otherwise
        returns None.
        """
        xp = self.main.get_xp_from_sig(sig)
        return Sheep(xp)

    def get_sheep_from_job_id(self, job_id: str) -> tp.Optional[Sheep]:
        """
        Returns the `Sheep` associated with the given `job_id`. If no sheep
        is found, returns None.
        """
        link = self._by_id / job_id
        if link.is_symlink():
            sig = link.resolve().name
            xp = self.main.get_xp_from_sig(sig)
            return Sheep(xp)
        return None

    def update(self):
        """
        Force an update of all job states with submitit.
        """
        SlurmJob.watcher.update()

    @contextmanager
    def job_array(self, slurm_config: SlurmConfig):
        """Context manager to launch XP in job array."""
        assert not self._in_job_array
        self._to_submit.append(_JobArray(slurm_config))
        self._in_job_array = True
        try:
            yield
        finally:
            self._in_job_array = False

    def maybe_submit_lazy(self, sheep: Sheep, slurm_config: SlurmConfig, rules: SubmitRules):
        """
        Decides whether to schedule a new job for the given sheep, based on the rules
        given in `rules`.
        Jobs are actually only scheduled once the `commit()` method is called.
        """
        if sheep.job is not None:
            state = sheep.state()
            if state == 'COMPLETED':
                if rules.replace_done:
                    logger.debug(f"Ignoring previously completed job {sheep.job.job_id}")
                    sheep.job = None
            elif state in ["FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT", "MISSING",
                           "NODE_FAIL"]:
                logger.debug(f"Previous job {sheep.job.job_id} failed or was canceled")
                if rules.retry:
                    sheep.job = None
            else:
                if rules.replace:
                    logger.debug(f"Cancelling previous job {sheep.job.job_id} with status {state}")
                    self.cancel_lazy(sheep.job)
                    sheep.job = None

        if sheep.job is None:
            if not self._in_job_array:
                self._to_submit.append(_JobArray(slurm_config))
            assert slurm_config == self._to_submit[-1].slurm_config
            self._to_submit[-1].sheeps.append(sheep)

    def cancel_lazy(self, job: submitit.SlurmJob):
        """
        Cancel a job. The job is actually cancelled only when `commit()` is called.
        """
        self._to_cancel.append(job)

    def commit(self):
        """
        Commit all changes registered so far with either `maybe_submit_lazy()`
        and `cancel_lazy()`.
        """
        if self._to_cancel:
            self._cancel(self._to_cancel)
            self._to_cancel = []

        self._existing_git_clone = None
        while self._to_submit:
            job_array = self._to_submit.pop(0)
            self._submit(job_array)

    @property
    def _by_id(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.by_id

    @property
    def _orphans(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.orphans

    @property
    def _arrays(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.arrays

    def _cancel(self, jobs: tp.List[SlurmJob]):
        cancel_cmd = ["scancel"] + [job.job_id for job in jobs]
        logger.debug("Running %s", " ".join(cancel_cmd))
        sp.run(cancel_cmd, check=True)

    def _get_submitit_executor(self, name: str, folder: Path,
                               slurm_config: SlurmConfig) -> submitit.SlurmExecutor:
        os.environ['SLURM_KILL_BAD_EXIT'] = '1'  # Kill the job if any of the task fails
        kwargs = dict(slurm_config.__dict__)
        executor = submitit.SlurmExecutor(
            folder=folder, max_num_timeout=kwargs.pop('max_num_timeout'))
        gpus = slurm_config.gpus
        if gpus > 8:
            if gpus % 8 != 0:
                raise ValueError("Can only take <= 8 gpus, or multiple of 8 gpus")
            kwargs['nodes'] = gpus // 8
            gpus_per_node = 8
        else:
            gpus_per_node = gpus
            kwargs['nodes'] = 1
        mem_per_gpu = slurm_config.mem_per_gpu
        if mem_per_gpu:
            mem = slurm_config.mem_per_gpu * gpus_per_node
            kwargs['mem'] = f"{mem}GB"
        kwargs['gres'] = f'gpu:{gpus_per_node}'
        if slurm_config.one_task_per_node:
            kwargs['ntasks_per_node'] = 1
            if slurm_config.cpus_per_task is None:
                kwargs['cpus_per_task'] = gpus_per_node * slurm_config.cpus_per_gpu
        else:
            kwargs['ntasks_per_node'] = gpus_per_node
            if slurm_config.cpus_per_task is None:
                kwargs['cpus_per_task'] = slurm_config.cpus_per_gpu
        del kwargs['gpus']
        del kwargs['mem_per_gpu']
        del kwargs['cpus_per_gpu']
        del kwargs['one_task_per_node']
        logger.debug("Slurm parameters %r", kwargs)

        executor.update_parameters(
            job_name=name,
            stderr_to_stdout=True,
            **kwargs)
        return executor

    def _check_orphans(self):
        """Check for orphaned jobs."""
        for dirty in self._orphans.iterdir():
            name = dirty.name
            logger.warning(f"Found dirty tag {name}, meaning a job might have been scheduled "
                           "but Dora or Slurm crashed before the job id was saved.")
            proc = sp.run(["squeue", "-u", os.getlogin(), "-n", name, "-o", "%i", "-h"],
                          capture_output=True, check=True)
            ids = [line for line in proc.stdout.decode().strip().split("\n") if line]
            if ids:
                logger.warning(f"Found orphan job ids {ids}, will cancel")
                sp.run(["scancel"] + ids, check=True)
            dirty.unlink()

    @contextmanager
    def _enter_orphan(self, name: str):
        """Context manager to enter a potential orphan."""
        token = self._orphans / name
        token.touch()
        try:
            yield
        finally:
            token.unlink()

    def _submit(self, job_array: _JobArray):
        sheeps = job_array.sheeps
        slurm_config = job_array.slurm_config
        if not sheeps:
            return

        is_array = len(sheeps) > 1
        first = sheeps[0]
        self.main.init_xp(first.xp)
        use_git_save = first.xp.dora.git_save
        assert all(other.xp.dora.git_save == use_git_save for other in sheeps), \
            "All jobs inside an array must have the same value for git_save."""

        if is_array:
            name_sig = _get_sig(sorted([sheep.xp.sig for sheep in sheeps]))
        else:
            name_sig = first.xp.sig
        if is_array:
            name = self.main.name + "_array_" + name_sig
        else:
            name = self.main.name + "_" + name_sig

        if is_array:
            submitit_folder = self._arrays / name
        else:
            submitit_folder = first.xp._xp_submitit
        submitit_folder.mkdir(exist_ok=True)

        for sheep in sheeps:
            xp = sheep.xp
            self.main.init_xp(xp)
            if xp.rendezvous_file.exists():
                xp.rendezvous_file.unlink()

        executor = self._get_submitit_executor(name, submitit_folder, slurm_config)
        jobs: tp.List[submitit.Job] = []
        if use_git_save and self._existing_git_clone is None:
            self._existing_git_clone = git_save.get_new_clone(self.main)
        with self._enter_orphan(name):
            with ExitStack() as stack:
                if use_git_save:
                    assert self._existing_git_clone is not None
                    stack.enter_context(git_save.enter_clone(self._existing_git_clone))
                if is_array:
                    stack.enter_context(executor.batch())
                for sheep in job_array.sheeps:
                    if use_git_save:
                        assert self._existing_git_clone is not None
                        git_save.assign_clone(sheep.xp, self._existing_git_clone)
                    jobs.append(executor.submit(_SubmitItTarget(), self.main, sheep.xp.argv))
            # Now we can access jobs
            for sheep, job in zip(sheeps, jobs):
                # See commment in `Sheep.state` function above for storing all jobs in the array.
                pickle.dump((job, jobs), open(sheep._job_file, "wb"))
                logger.debug("Created job with id %s", job.job_id)
                sheep.job = job  # type: ignore
                sheep._other_jobs = jobs  # type: ignore
                link = self._by_id / job.job_id
                link = link
                link.symlink_to(sheep.xp.folder.resolve())
                if is_array:
                    # We link the array submitit folder to be sure
                    # we keep an history of all arrays the XP was in.
                    submitit_link = (sheep.xp.folder / submitit_folder.name)
                    if submitit_link.exists():
                        assert submitit_link.resolve() == submitit_folder.resolve()
                    else:
                        submitit_link.symlink_to(submitit_folder)
                latest = sheep.xp._latest_submitit
                if latest.exists():
                    latest.unlink()
                latest.symlink_to(submitit_folder)

                name = self.main.get_name(sheep.xp)
                self.log(f"Scheduled job {job.job_id} for sheep {sheep.xp.sig}/{name}")
