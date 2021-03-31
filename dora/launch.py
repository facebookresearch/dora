"""
"""
from functools import partial
import subprocess as sp
import time

from .conf import SubmitRules, update_from_args
from .main import DecoratedMain
from .shep import Shepherd
from .log import simple_log

log = partial(simple_log, "Launch:")


def launch_action(args, main: DecoratedMain):
    shepherd = Shepherd(main, log=log)
    slurm = main.get_slurm_config()
    update_from_args(slurm, args)
    rules = SubmitRules()
    update_from_args(rules, args)

    sheep = shepherd.get_sheep(args.argv)
    log(f"Fetched sheep {sheep}")
    shepherd.update()
    shepherd.maybe_submit_lazy(sheep, slurm, rules)
    shepherd.commit()

    if args.tail or args.attach:
        done = False
        tail_process = None
        try:
            while True:
                if sheep.log.exists() and tail_process is None:
                    tail_process = sp.Popen(["tail", "-n", "200", "-f", sheep.log])
                if sheep.is_done("force"):
                    log("Remote process finished with state", sheep.state())
                    done = True
                    break
                time.sleep(30)
        except KeyboardInterrupt:
            log("KeyboardInterrupt received...")
        finally:
            if tail_process:
                tail_process.kill()
            if args.attach and not done:
                log(f"attach is set, killing remote job {sheep.job.job_id}")
                sheep.job.cancel()
