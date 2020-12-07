import logging
from pathlib import Path
import pickle
import os
import shutil
import sys


from submitit import SlurmJob, JobEnvironment
import submitit

from .utils import try_load


logger = logging.getLogger(__name__)


class SubmitItTarget:
    def __call__(self, main, argv):
        env = JobEnvironment()
        rank = env.global_rank
        world_size = env.num_tasks
        os.environ['DORA_RANK'] = str(rank)
        os.environ['DORA_WORLD_SIZE'] = str(world_size)
        sys.argv[1:] = argv
        main()

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


class Sheep:
    def __init__(self, cfg, overrides):
        self.cfg = cfg
        self.overrides = overrides
        self.job = None
        if self.job_file.exists():
            self.job = try_load(self.job_file, pickle.load)

    @property
    def folder(self):
        return Path(self.cfg.hydra.run.dir)

    @property
    def log(self):
        return self.folder / self.cfg.hydra.job_logging.handlers.file.filename

    @property
    def job_file(self):
        return self.folder / self.cfg.dora.job_file

    @property
    def sig(self):
        return self.cfg.dora.sig

    def is_done(self):
        return SlurmJob.watcher.is_done(self.job.job_id, 'force')

    @property
    def state(self):
        return None if self.job is None else self.job.state

    def __repr__(self):
        return f"Sheep({self.sig}, state={self.state}, overrides={self.overrides})"


class Shepherd:
    """
    Takes care of the little jobs.
    """
    def __init__(self, main):
        self.main = main
        self.cfg = self.main.get_config(return_hydra_config=True)
        self.dbs.mkdir(exist_ok=True, parents=True)
        (self.dbs / "job_ids").mkdir(exist_ok=True)

    def get_sheep(self, overrides):
        cfg = self.main.get_config(overrides, return_hydra_config=True)
        return Sheep(cfg, overrides)

    def update(self):
        SlurmJob.watcher.update()

    def submit(self, sheep):
        folder = sheep.folder / sheep.cfg.dora.submitit
        if folder.exists():
            shutil.rmtree(folder)
        rendezvous_file = sheep.folder / sheep.cfg.dora.ddp.rendezvous_file
        if rendezvous_file.exists():
            rendezvous_file.unlink()
        if sheep.log.exists():
            sheep.log.unlink()

        # Kill the job if any of the task fails
        os.environ['SLURM_KILL_BAD_EXIT'] = '1'
        executor = submitit.SlurmExecutor(folder=folder)
        slurm = sheep.cfg.slurm
        gpus = slurm.gpus
        if gpus > 8:
            if gpus % 8 != 0:
                raise RuntimeError("Can only take <= 8 gpus, or multiple of 8 gpus")
            slurm.nodes = gpus // 8
            slurm.ntasks_per_node = 8
        else:
            slurm.ntasks_per_node = gpus
            slurm.nodes = 1
        mem = slurm.mem_per_task * slurm.ntasks_per_node
        slurm.mem = f"{mem}GB"
        slurm = dict(slurm)
        del slurm['gpus']
        del slurm['mem_per_task']
        logger.debug("Slurm parameters %r", slurm)

        name = self.module + ":" + sheep.cfg.dora.sig
        executor.update_parameters(job_name=name, **slurm)
        job = executor.submit(
            SubmitItTarget(), self.main, sheep.overrides)
        logger.info(f'Scheduled using Submitit, Job ID: {job.job_id}')
        pickle.dump(job, open(sheep.job_file, "wb"))
        sheep.job = job
        link = self.dbs / "job_ids" / job.job_id
        link = link
        link.symlink_to(sheep.folder.resolve())

    @property
    def dbs(self):
        return Path(self.cfg.dora.dbs)

    def get_sheep_from_jid(self, job_id):
        link = self.dbs / "job_ids" / job_id
        if link.is_symlink():
            sig = link.resolve().name
            cfg = self.main.get_config_from_sig(sig, return_hydra_config=True)
            overrides = self.main.get_overrides_from_sig(sig)
            return Sheep(cfg, overrides)
        return None
