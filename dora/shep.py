import logging
from pathlib import Path
import pickle
import os
import subprocess as sp
import sys
import typing as tp


from submitit import SlurmJob, JobEnvironment
import submitit

from .conf import SlurmConfig, SubmitRules
from .main import DecoratedMain
from .utils import try_load
from .xp import XP


logger = logging.getLogger(__name__)


class SubmitItTarget:
    def __call__(self, main: DecoratedMain, argv: tp.Sequence[str]):
        env = JobEnvironment()
        rank = env.global_rank
        world_size = env.num_tasks
        os.environ['LOCAL_RANK'] = str(env.local_rank)
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        sys.argv[1:] = argv
        main()

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


class Sheep:
    """
    A Sheep is a specific run for a given XP. Sheeps are managed
    by the Shepherd.
    """
    def __init__(self, xp: XP, job: SlurmJob = None):
        self.xp = xp
        self.job: tp.Optional[submitit.SlurmJob] = None
        if self.job_file.exists():
            self.job = try_load(self.job_file, load=pickle.load)

    @property
    def job_file(self) -> Path:
        return self.xp.folder / self.xp.dora.shep.job_file

    def state(self, mode="standard"):
        if self.job is None:
            return None
        state = self.job.watcher.get_state(self.job.job_id, mode)
        if state.startswith('CANCELLED'):
            return 'CANCELLED'
        return state

    def is_done(self, mode="standard"):
        if self.job is None:
            return False
        return self.job.watcher.is_done(self.job.job_id, mode)

    @property
    def log(self):
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
    pass


class Shepherd:
    """
    Takes care of the little jobs.
    """
    def __init__(self, main: DecoratedMain, log: tp.Callable[[str], None] = no_log):
        self.main = main
        self.by_id.mkdir(exist_ok=True, parents=True)
        self.grids.mkdir(exist_ok=True, parents=True)
        self.log = log

        self._to_cancel = []
        self._to_submit = []

    def get_sheep(self, argv: tp.Sequence[str]):
        xp = self.main.get_xp(argv)
        return Sheep(xp)

    def update(self):
        SlurmJob.watcher.update()

    @property
    def by_id(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.by_id

    @property
    def grids(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.grids

    def maybe_submit_lazy(self, sheep: Sheep, slurm_config: SlurmConfig, rules: SubmitRules):
        if sheep.job is not None:
            state = sheep.state()
            if state == 'COMPLETED':
                if rules.replace_done:
                    self.log(f"Ignoring previously completed job {sheep.job.job_id}")
                    sheep.job = None
            elif state in ["FAILED", "CANCELLED"]:
                self.log(f"Previous job {sheep.job.job_id} failed or was canceled")
                if rules.retry:
                    sheep.job = None
            else:
                if rules.replace:
                    self.log(f"Cancelling previous job {sheep.job.job_id} with status {state}")
                    self.cancel_lazy(sheep)
                    sheep.job = None

        if sheep.job is None:
            self._to_submit.append((sheep, slurm_config))

    def cancel_lazy(self, sheep: Sheep):
        self._to_cancel.append(sheep)

    def commit(self):
        if self._to_cancel:
            cancel_cmd = ["scancel"] + [s.job.job_id for s in self._to_cancel]
            sp.run(cancel_cmd, check=True)
            self._to_cancel = []

        while self._to_submit:
            sheep, slurm_config = self._to_submit.pop(0)
            self._submit(sheep, slurm_config)
            self.log(f"Job created with id {sheep.job.job_id}")

    def _submit(self, sheep, slurm_config: SlurmConfig):
        xp = sheep.xp
        self.main.init_xp(xp)
        folder = xp.folder / xp.dora.shep.submitit_folder
        if xp.rendezvous_file.exists():
            xp.rendezvous_file.unlink()

        # Kill the job if any of the task fails
        os.environ['SLURM_KILL_BAD_EXIT'] = '1'
        kwargs = dict(slurm_config.__dict__)
        executor = submitit.SlurmExecutor(
            folder=folder, max_num_timeout=kwargs.pop('max_num_timeout'))
        gpus = slurm_config.gpus
        if gpus > 8:
            if gpus % 8 != 0:
                raise RuntimeError("Can only take <= 8 gpus, or multiple of 8 gpus")
            kwargs['nodes'] = gpus // 8
            kwargs['ntasks_per_node'] = 8
        else:
            kwargs['ntasks_per_node'] = gpus
            kwargs['nodes'] = 1
        mem = slurm_config.mem_per_gpu * kwargs['ntasks_per_node']
        kwargs['mem'] = f"{mem}GB"
        del kwargs['gpus']
        del kwargs['mem_per_gpu']
        logger.debug("Slurm parameters %r", kwargs)

        name = self.main.name + ":" + sheep.xp.sig
        executor.update_parameters(
            job_name=name,
            stderr_to_stdout=True,
            **kwargs)
        job = executor.submit(
            SubmitItTarget(), self.main, sheep.xp.argv)
        pickle.dump(job, open(sheep.job_file, "wb"))
        logger.debug("Created job with id %s", job.job_id)
        sheep.job = job
        link = self.by_id / job.job_id
        link = link
        link.symlink_to(sheep.xp.folder.resolve())

    def get_sheep_from_job_id(self, job_id: str) -> tp.Optional[Sheep]:
        link = self.dbs / "job_ids" / job_id
        if link.is_symlink():
            sig = link.resolve().name
            xp = self.main.get_xp_from_sig(sig)
            return Sheep(xp)
        return None
