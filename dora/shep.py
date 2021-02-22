import logging
from pathlib import Path
import pickle
import os
import sys
import typing as tp


from submitit import SlurmJob, JobEnvironment
import submitit

from .conf import DoraRun, SlurmConfig
from .main import DecoratedMain
from .utils import try_load


logger = logging.getLogger(__name__)


class SubmitItTarget:
    def __call__(self, main: DecoratedMain, argv: tp.Sequence[str]):
        env = JobEnvironment()
        rank = env.global_rank
        world_size = env.num_tasks
        os.environ['LOCAL_RANK'] = env.local_rank
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        sys.argv[1:] = argv
        main()

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


class Sheep:
    def __init__(self, run: DoraRun):
        self.run = run
        self.job: tp.Optional[submitit.SlurmJob] = None
        if self.job_file.exists():
            self.job = try_load(self.job_file, pickle.load)

    @property
    def job_file(self) -> Path:
        return self.folder / self.cfg.dora.job_file

    @property
    def state(self):
        if self.job is None:
            return None
        state = self.job.state
        if state.startswith('CANCELED'):
            return 'CANCELED'
        return state

    def __repr__(self):
        return f"Sheep({self.run.sig}, state={self.state}, argv={self.run.argv})"


class Shepherd:
    """
    Takes care of the little jobs.
    """
    def __init__(self, main: DecoratedMain):
        self.main = main
        self.by_id.mkdir(exist_ok=True, parents=True)

    def get_sheep(self, argv: tp.Sequence[str]):
        run = self.main.get_run(argv)
        return Sheep(run)

    def update(self):
        SlurmJob.watcher.update()

    @property
    def by_id(self) -> Path:
        return self.main.dora.dir / self.main.dora.shep.by_id

    def submit(self, sheep, slurm_config: SlurmConfig):
        run = sheep.run
        folder = run.folder / run.dora.shep.submitit_folder
        if run.rendezvous_file.exists():
            run.rendezvous_file.unlink()

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

        name = self.module + ":" + sheep.cfg.dora.sig
        executor.update_parameters(job_name=name, **kwargs)
        job = executor.submit(
            SubmitItTarget(), self.main, sheep.overrides)
        pickle.dump(job, open(sheep.job_file, "wb"))
        sheep.job = job
        link = self.by_id / job.job_id
        link = link
        link.symlink_to(sheep.run.folder.resolve())

    def get_sheep_from_job_id(self, job_id: str) -> Sheep:
        link = self.dbs / "job_ids" / job_id
        if link.is_symlink():
            sig = link.resolve().name
            run = self.main.get_run_from_sig(sig)
            return Sheep(run)
        return None
