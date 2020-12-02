from functools import partial
import subprocess as sp
import time

from .shep import Shepherd
from .log import simple_log

log = partial(simple_log, "Launch:")


def _state_machine(args, sheep):
    if sheep.job is not None:
        if sheep.job.state.startswith("CANCELLED") or sheep.job.state == "FAILED":
            log(f"Previous job {sheep.job.job_id} failed or was canceled")
            sheep.job = None
            return
        if args.replace:
            if sheep.job.state == "COMPLETED":
                log(f"Ignoring previously completed job {sheep.job.job_id}")
            else:
                log(f"Cancelling previous job {sheep.job.job_id} and status {sheep.job.state}")
                sheep.job.cancel()
            sheep.job = None
        else:
            log(f"Found previous job {sheep.job.job_id} with status {sheep.job.state}")


def launch_action(args, hydra_support, module):
    if args.dev:
        args.attach = True
        args.partition = "dev"
    for name in ["gpus", "partition", "comment"]:
        value = getattr(args, name)
        if value is not None:
            args.overrides.append(f"slurm.{name}={value}")
    shepherd = Shepherd(hydra_support, module)
    sheep = shepherd.get_sheep(args.overrides)
    log(f"Fetched sheep {sheep}")
    _state_machine(args, sheep)

    if sheep.job is None:
        shepherd.submit(sheep)
        log(f"Job created with id {sheep.job.job_id}")

    if args.tail or args.attach:
        done = False
        tail_process = None
        try:
            while True:
                if sheep.log.exists():
                    tail_process = sp.Popen(["tail", "-n", "200", "-f", sheep.log])
                if sheep.is_done():
                    if sheep.log.exists():
                        tail_process = sp.Popen(["tail", "-n", "200", "-f", sheep.log])
                    log("Remote process finished with state", sheep.state)
                    done = True
                    break
                time.sleep(30)
        finally:
            if tail_process:
                tail_process.kill()
            if not done:
                log(f"attach is set, killing remote job {sheep.job.job_id}")
                sheep.job.cancel()
