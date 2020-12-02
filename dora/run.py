import os
import sys

from .executor import start_ddp_workers


def run_action(args, hydra_support, module):
    if args.ddp:
        start_ddp_workers(module, hydra_support, args.overrides)
    else:
        os.execvp(sys.executable, [sys.executable, "-m", module])
