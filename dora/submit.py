# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Alexandre Défossez @adefossez, 2020
"""
Submission utilities, recover jobs and submit new ones.
"""

import os
import logging
import pickle
import shutil
import subprocess as sp
import sys

import hydra
import submitit
from submitit.slurm.slurm import SlurmJob


from .utils import swap_cwd

logger = logging.getLogger(__name__)


def get_existing_job(args):
    job = None
    if os.path.exists(args.job_file):
        job = load_job(args.job_file)
    return job


def load_job(path):
    job = pickle.load(open(path, "rb"))
    SlurmJob.watcher.register_job(job.job_id)
    return job


def save_job(job, path):
    pickle.dump(job, open(path, "wb"))


def submit_job(name, args):
    folder = os.path.abspath("./submitit")
    if args.clean and os.path.exists(folder):
        logger.warning("Clean is on, removing %s", folder)
        shutil.rmtree(folder)
    if os.path.exists(args.rendezvous_file):
        os.unlink(args.rendezvous_file)
    if os.path.exists(args.remote_log):
        os.unlink(args.remote_log)
    # using submitit, hydra already got us a folder so let's use it
    # go back to the previous working directory
    with swap_cwd(hydra.utils.get_original_cwd()):
        os.environ['SLURM_KILL_BAD_EXIT'] = '1'
        executor = submitit.SlurmExecutor(folder=folder)
        slurm = args.slurm
        gpus = args.gpus
        if gpus > 8:
            if gpus % 8 != 0:
                logger.fatal("Can only take <= 8 gpus, or multiple of 8 gpus")
                sys.exit(1)
            slurm.nodes = gpus // 8
            slurm.ntasks_per_node = 8
        else:
            slurm.ntasks_per_node = gpus
        mem = args.mem_gb * slurm.ntasks_per_node
        slurm.mem = f"{mem}GB"
        if args.dev:
            slurm.partition = 'dev'
        logger.debug("Slurm parameters %r", slurm)
        executor.update_parameters(job_name=name, **slurm)
        job = executor.submit(SubmitItTarget(), args.remote_log, sys.argv)
        logger.info(f'Scheduled using Submitit, Job ID: {job.job_id}')
    save_job(job, args.job_file)
    return job


class SubmitItTarget:
    def __call__(self, log, argv):
        from . import distrib
        distrib.init_rank()
        if distrib.rank > 0:
            log += f".{distrib.rank}"
        extra = [
            "hydra.job_logging.handlers.file.filename=" + log, "scheduled=1"]
        sp.run([sys.executable] + argv + extra)

    def checkpoint(self, *args, **kwargs):
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)