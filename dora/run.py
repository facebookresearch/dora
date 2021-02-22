import os
import sys

from .executor import start_ddp_workers


def run_action(args, main):
    if args.ddp and not os.environ.get('RANK'):
        start_ddp_workers(args.package, main, args.argv)
    else:
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
        sys.argv[1:] = args.argv
        main()
