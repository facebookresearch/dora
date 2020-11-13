#!/usr/bin/env python3
# Created on 2020/03
# Author: Yossi Adi (adiyoss)
# Contributors: Alexandre Défossez (adefossez)

import json
import logging
import os
import socket
import subprocess as sp
import sys
import time

import hydra


logger = logging.getLogger(__name__)


def run(args):
    import torch
    from tensorboardX import SummaryWriter

    from denoise import distrib
    from denoise.data.data import AudioDataset, Trainset, Validset
    from denoise.models.demucs import Demucs
    from denoise.models.mulcat_gate import DenoiseWave
    from denoise.models.mulcat_gate_sym import DenoiseWaveSym
    from denoise.solver import Solver
    logger.info("Running on host %s", socket.gethostname())
    distrib.init(args.rendezvous_file)

    if args.model == "demucs":
        model = Demucs(**args.demucs)
    elif args.model == "dwave":
        kwargs = dict(args.dwave)
        sym = kwargs.pop("sym")
        kwargs['sr'] = args.sample_rate
        kwargs['segment'] = args.segment
        if sym:
            model = DenoiseWaveSym(**kwargs)
        else:
            model = DenoiseWave(**kwargs)
    else:
        logger.fatal("Invalid model name %s", args.model)
        os._exit(1)

    if args.trains:
        from trains import Task
        Task.set_credentials(host='http://100.97.67.41:8008', key='NIBC3PASB9TY06IFQZ8M',
                             secret='DI2LzcL!p8%rSbQv#W+mv)fue7tr05yWwOIg8dQtnePXGa!gt1')
        Task.init(
            project_name="SpeechEnhancement", task_name=os.path.basename(args.save_folder))
        writer = SummaryWriter('runs')
    else:
        writer = None

    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        segment_len = int(args.segment * args.sample_rate)
        segment_len = model.valid_length(segment_len)
        args.segment = segment_len / args.sample_rate

    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)

        return

    assert args.batch_size % distrib.world_size == 0
    args.batch_size //= distrib.world_size

    # Building datasets and loaders
    if args.stride:
        tr_dataset = Trainset(
            args.dset.train, sample_rate=args.sample_rate, segment=args.segment,
            stride=args.stride, pad=args.pad, noise_path=args.dset.noise_path,
            aug_prob=args.aug_prob, dymix_prob=args.dymix_prob, reverb=args.reverb,
            rev_strong=args.rev_strong,
            custom=args.custom_reverb, two_sources=args.two_sources, rev_noise=args.rev_noise,
            rev_clean=args.rev_clean)
        tr_loader = distrib.loader(
            tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        tr_dataset = AudioDataset(
            args.dset.train, args.batch_size,
            sample_rate=args.sample_rate, segment=args.segment)
        tr_loader = distrib.audio_loader(
            tr_dataset, batch_size=1, shuffle=True,
            num_workers=args.num_workers)

    # segment=-1 -> use full audio # batch_size=1 -> use less GPU memory to do cv
    if args.validfull:
        cv_dataset = Validset(args.dset.valid, args.sample_rate)
        tt_dataset = Validset(args.dset.test, args.sample_rate)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_dataset = AudioDataset(
            args.dset.valid, batch_size=1,
            sample_rate=args.sample_rate,
            segment=-1, cv_maxlen=args.cv_maxlen)
        tt_dataset = AudioDataset(
            args.dset.test, batch_size=args.batch_size,
            sample_rate=args.sample_rate,
            segment=-1, cv_maxlen=args.cv_maxlen)
        cv_loader = distrib.audio_loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
        tt_loader = distrib.audio_loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    # torch also initialize cuda seed if available
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = Solver(data, model, optimizer, writer, args)
    solver.train()


def _main(args):
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str):
            args.dset[key] = hydra.utils.to_absolute_path(value)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    if args.continue_from:
        args.continue_from = os.path.join(
            os.getcwd(), "..", args.continue_from, args.checkpoint_file)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)

    name = os.path.split(os.getcwd())[-1]
    if args.name:
        json.dump({'name': name, 'folder': os.getcwd()}, sys.stdout)
        return

    if args.dev:
        args.submit = 1
        args.restart = 1
    if not args.scheduled and (args.submit or args.cancel or args.query or args.check):
        from denoise.submit import get_existing_job, submit_job
        job = get_existing_job(args)
        state = None
        running = False
        done = False
        if job:
            state = job.state
            if state == 'COMPLETED':
                state = 'DONE'  # rename to avoid clash when showing only prefix
                done = True
            running = not job.watcher.is_done(job.job_id)
            logger.info("Found previous job %s with state %s", job.job_id, state)
        if args.cancel:
            if running:
                logger.warning("Canceling existing job %s", job.job_id)
                job.cancel()
                job = None
            else:
                logger.warning("Asking to cancel, but job does not exist")

        if args.submit:
            submit = True
            if done:
                if not args.replace_done:
                    submit = False
                    logger.info("Job has already completed. "
                                "Use replace_done=1 to force rescheduling.")
            elif running:
                if args.replace:
                    logger.info("Replacing job, previous job will be canceled.")
                    job.cancel()
                elif args.replace_pending and state == "PENDING":
                    logger.info("Previous job has not yet started yet, and i"
                                "replace_pending=1, replacing it.")
                    job.cancel()
                else:
                    submit = False
            elif state and (state == "FAILED" or state.startswith("CANCELLED")):
                if args.retry:
                    logger.info('Previous job failed and retry=1, scheduling new job.')
                else:
                    logger.info('Previous job failed, set retry=1 to schedule new job.')
                    submit = False
            if submit:
                job = submit_job(name, args)
                state = "SCHEDULED"
                running = True

        if args.query:
            sid = None
            if job:
                sid = job.job_id
            try:
                history = json.load(open(args.history_file, "rb"))
            except IOError:
                history = None
            json.dump({"sid": sid, "state": state, "history": history}, sys.stdout)
        if args.tail:
            if not running:
                try:
                    print(open(args.remote_log, "r").read())
                except IOError:
                    pass
                return
            if not job:
                return
            tail_process = None
            try:
                while True:
                    if tail_process is None and os.path.exists(args.remote_log):
                        logger.info("Remote job started, tailing log")
                        tail_process = sp.Popen(["tail", "-n", "+0", "-F", args.remote_log])
                    if job.watcher.is_done(job.job_id, 'force'):
                        job = None
                        break
                    time.sleep(30)
            finally:
                if tail_process:
                    tail_process.terminate()
                if not args.check and args.kill_if_tail and job:
                    logger.warning("kill_if_tail set, killing job %s", job.job_id)
                    job.cancel()

    else:
        run(args)


@hydra.main(config_path="conf/config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        os._exit(1)  # a bit dangerous for the logs, but Hydra intercepts exit code
        # fixed in beta but I could not get the beta to work


if __name__ == "__main__":
    main()