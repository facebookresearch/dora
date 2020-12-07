import os
import sys

from .executor import start_ddp_workers


def run_action(args, main):
    if args.ddp and not os.environ.get('DORA_CHILD'):
        start_ddp_workers(args.package, main, args.overrides)
    else:
        os.environ['DORA_RANK'] = '0'
        os.environ['DORA_WORLD_SIZE'] = '1'
        sys.argv[1:] = args.overrides
        main()
