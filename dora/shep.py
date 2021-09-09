import logging
from pathlib import Path
import pickle
import os
import subprocess as sp
import sys
import typing as tp


from submitit import SlurmJob
import submitit

from .conf import SlurmConfig, SubmitRules
from .distrib import get_distrib_spec
from .git_save import git_save
from .main import DecoratedMain
from .utils import try_load
from .xp import XP


logger = logging.getLogger(__name__)


class _SubmitItTarget:
    def __call__(self, main: DecoratedMain, argv: tp.Sequence[str]):
        self.xp = main.get_xp(argv)
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
    def __init__(self, xp: XP, job: SlurmJob = None):
        self.xp = xp
        self.job: tp.Optional[submitit.SlurmJob] = None
        if self._job_file.exists():
            self.job = try_load(self._job_file)

    @property
    def _job_file(self) -> Path:
        return self.xp.folder / self.xp.dora.shep.job_file

    def state(self, mode="standard"):
        """Return the current state of the `Sheep`.
        """
        if self.job is None:
            return None
        state = self.job.watcher.get_state(self.job.job_id, mode)
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
        self.log = log

        self._to_cancel: tp.List[submitit.SlurmJob] = []
        self._to_submit: tp.List[tp.Tuple[Sheep, SlurmConfig]] = []

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
            elif state in ["FAILED", "CANCELLED", "OUT_OF_MEMORY", "TIMEOUT"]:
                logger.debug(f"Previous job {sheep.job.job_id} failed or was canceled")
                if rules.retry:
                    sheep.job = None
            else:
                if rules.replace:
                    logger.debug(f"Cancelling previous job {sheep.job.job_id} with status {state}")
                    self.cancel_lazy(sheep.job)
                    sheep.job = None

        if sheep.job is None:
            self._to_submit.append((sheep, slurm_config))

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

        while self._to_submit:
            sheep, slurm_config = self._to_submit.pop(0)
            self._submit(sheep, slurm_config)
            name = self.main.get_name(sheep.xp)
            self.log(f"Scheduled job {sheep.job.job_id} for sheep {sheep.xp.sig}/{name}")

    @property
    def _by_id(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.by_id

    def _cancel(self, jobs: tp.List[SlurmJob]):
        cancel_cmd = ["scancel"] + [job.job_id for job in self._to_cancel]
        logger.debug("Running %s", " ".join(cancel_cmd))
        sp.run(cancel_cmd, check=True)

    def _submit(self, sheep, slurm_config: SlurmConfig):
        xp = sheep.xp
        self.main.init_xp(xp)
        folder = xp.folder / xp.dora.shep.submitit_folder
        folder.mkdir(exist_ok=True)
        name = self.main.name + "_" + sheep.xp.sig

        # Dirty flag is used to detect if a job was submitted, but Dora was
        # killed before saving its job id.
        dirty = folder / "dirty"
        if dirty.exists():
            # Previous tentative for submission might have failed with an orphan job.
            # We manually scan for a job with the right name.
            long_name = self.main.get_name(sheep.xp)
            logger.warning("Dirty tag is still here, we might have an orphan job "
                           f"for sheep {sheep.xp.sig}/{long_name}.")
            proc = sp.run(["squeue", "-n", name, "-o", "%i", "-h"],
                          capture_output=True, check=True)
            ids = [line for line in proc.stdout.decode().strip().split("\n") if line]
            if ids:
                logger.warning(f"Found orphan job ids {ids}, will cancel")
                sp.run(["scancel"] + ids, check=True)

        if xp.rendezvous_file.exists():
            xp.rendezvous_file.unlink()

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
        mem = slurm_config.mem_per_gpu * gpus_per_node
        kwargs['mem'] = f"{mem}GB"
        if slurm_config.one_task_per_node:
            kwargs['gpus_per_task'] = gpus_per_node
            kwargs['ntasks_per_node'] = 1
            if slurm_config.cpus_per_task is None:
                kwargs['cpus_per_task'] = gpus_per_node * slurm_config.cpus_per_gpu
        else:
            kwargs['gpus_per_task'] = 1
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

        with git_save(xp, xp.dora.git_save):
            dirty.touch()
            job = executor.submit(
                _SubmitItTarget(), self.main, sheep.xp.argv)

        pickle.dump(job, open(sheep._job_file, "wb"))
        logger.debug("Created job with id %s", job.job_id)
        sheep.job = job
        link = self._by_id / job.job_id
        link = link
        link.symlink_to(sheep.xp.folder.resolve())

        # Now we safely stored the job id, we can remove the dirty tag.
        dirty.unlink()
