import os
import sys

from .executor import start_ddp_workers
from .main import DecoratedMain


def run_action(args, main: DecoratedMain):
    if args.ddp and not os.environ.get('RANK'):
        start_ddp_workers(args.package, main, args.argv)
    else:
        if 'WORLD_SIZE' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
        sys.argv[1:] = args.argv
        main()
