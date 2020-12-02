import logging
from pathlib import Path
import pickle
import os
import shutil
import subprocess as sp
import sys


from submitit import SlurmJob
import submitit

from .utils import try_load


logger = logging.getLogger(__name__)


class SubmitItTarget:
    def __call__(self, log, argv):
        from . import distrib
        distrib.init_rank()
        if distrib.rank > 0:
            log += f".{distrib.rank}"
        extra = [
            "hydra.job_logging.handlers.file.filename=" + log]
        sp.run([sys.executable] + argv + extra)

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)


class Sheep:
    def __init__(self, cfg, overrides):
        self.cfg = cfg
        self.overrides = overrides
        if self.job_file.exists():
            self.job = try_load(self.job_file, pickle.load)

    @property
    def folder(self):
        return Path(self.cfg.hydra.run.dir)

    @property
    def log(self):
        return self.cfg.hydra.job_logging.handlers.file.filename

    @property
    def job_file(self):
        self.job_file = self.folder / self.cfg.dora.job_file

    def is_done(self):
        return SlurmJob.watcher.is_done(self.job.job_id, 'force')


class Shepherd:
    """
    Takes care of the little jobs.
    """
    def __init__(self, hydra_support, module):
        self.hydra_support = hydra_support
        self.module = module

    def get_sheep(self, overrides):
        cfg = self.hydra_support.get_config(overrides, return_hydra_config=True)
        return Sheep(cfg, overrides)

    def update(self):
        SlurmJob.watcher.update()

    def submit(self, sheep):
        folder = sheep.folder / sheep.cfg.dora.submitit
        if folder.exists():
            shutil.rmtree(folder)
        rendezvous = sheep.folder / sheep.cfg.dora.ddp.rendezvous
        if rendezvous.exists():
            rendezvous.unlink()
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
        del slurm['gpus']
        del slurm['mem_per_task']
        logger.debug("Slurm parameters %r", slurm)

        name = self.module + ":" + sheep.cfg.dora.sig
        executor.update_parameters(job_name=name, **slurm)
        job = executor.submit(SubmitItTarget(), sheep.log, ["-m", self.module] + sheep.overrides)
        logger.info(f'Scheduled using Submitit, Job ID: {job.job_id}')
        pickle.dump(job, open(sheep.job_file, "w"))
        sheep.job = job
        return job
