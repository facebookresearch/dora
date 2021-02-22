from functools import partial
import os
import shutil
import sys

from .shep import Shepherd
from .log import simple_log, fatal

log = partial(simple_log, "Info:")


def info_action(args, hydra_support, module):
    shepherd = Shepherd(hydra_support, module)
    if args.jid is not None:
        sheep = shepherd.get_sheep_from_jid(args.jid)
    else:
        sheep = shepherd.get_sheep(args.overrides)
    if sheep is None:
        fatal("Could not find any matching sheep")
    log("Found sheep", sheep)
    if sheep.job is not None:
        log("Job id is", sheep.job.job_id)
    log("Folder is", sheep.folder)
    log("Main log is", sheep.log)
    if args.cancel:
        if sheep.job is None:
            log("Could not cancel non existing job")
        elif sheep.is_done():
            log("Job is not running")
        else:
            sheep.job.cancel()
    if args.log:
        if not sheep.log.exists():
            fatal(f"Log {sheep.log} does not exist")
        shutil.copyfileobj(open(sheep.log, "r"), sys.stdout, 4096)
    if args.tail:
        if not sheep.log.exists():
            fatal(f"Log {sheep.log} does not exist")
        os.execvp("tail", ["tail", "-n", "200", "-f", args.log])
