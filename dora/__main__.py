"""
This is the central dispatch of the `dora` command. From there you can
check grid files, launch XPs, check their logs etc, as well
as doing local runs for debugging.
"""
import argparse
import logging
from pathlib import Path
import os
import sys

from .main import DecoratedMain
from .grid import grid_action
from .info import info_action
from .launch import launch_action
from .log import fatal, simple_log
from .run import run_action
from .utils import import_or_fatal


def _find_package():
    cwd = Path(".")
    candidates = []
    for child in cwd.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            if (child / "train.py").exists():
                candidates.append(child.name)
    if len(candidates) == 0:
        fatal("Could not find a training package. Use -P, or set DORA_PACKAGE.")
    elif len(candidates) == 1:
        return candidates[0]
    else:
        fatal(f"Found multiple candidates: {', '.join(candidates)}. "
              "Use -P, or set DORA_PACKAGE.")


def add_submit_rules(parser):
    parser.add_argument("-r", "--retry", action="store_true",
                        help="Retry failed jobs")
    parser.add_argument("-R", "--replace", action="store_true",
                        help="Replace any running job.")
    parser.add_argument("-D", "--replace_done", action="store_true",
                        help="Also resubmit done jobs.")


def add_slurm_config(parser):
    parser.add_argument("-g", "--gpus", type=int, help="Number of gpus.", default=1)
    parser.add_argument("-p", "--partition", default="learnfair", help="Partition.")
    parser.add_argument("--dev", action="store_const", dest="partition", const="dev",
                        help="Use dev partition.")
    parser.add_argument("-c", "--comment", help="Comment.")


def get_parser():
    parser = argparse.ArgumentParser()
    module = os.environ.get('DORA_PACKAGE')
    parser.add_argument(
        '--package', '-P',
        default=module,
        help='Training module.'
             'You can also set the DORA_PACKAGE env. In last resort, '
             'Dora will look for a package in the current folder with a train.py module.')
    parser.add_argument('--verbose', '-v', action='store_true', help="Show debug info.")
    subparsers = parser.add_subparsers(
        title="command", help="Command to execute", required=True, dest='command')
    grid = subparsers.add_parser("grid")
    add_submit_rules(grid)
    add_slurm_config(grid)
    grid.add_argument("-C", "--cancel", action='store_true',
                      help="Cancel all running jobs.")
    grid.add_argument("-i", "--interval", default=5, type=float,
                      help="Update status and metrics every that number of minutes. "
                           "Default is 5 min.")
    grid.add_argument("--no_monitoring", action="store_false", dest="monitor",
                      help="No monitoring, just schedule and print current state.")

    grid.add_argument("--dry_run", action="store_true",
                      help="Only simulate actions but does not run any call to Slurm.")
    grid.add_argument("-T", "--trim", type=int,
                      help="Trim history to the length of the exp with the given index.")
    grid.add_argument("-L", "--trim_last", action="store_true",
                      help="Trim history to the slowest.")

    group = grid.add_mutually_exclusive_group()
    group.add_argument("-f", "--folder", type=int,
                       help="Show the folder for the job with the given index")
    group.add_argument("-l", "--log", type=int,
                       help="Show the log for the job with the given index")
    group.add_argument("-t", "--tail", type=int,
                       help="Show the log for the job with the given index")

    grid.add_argument(
        'grid', nargs='?',
        help='Name of the grid to run. Name of the module will be `package`.grids.`name`.')

    grid.add_argument("patterns", nargs='*',
                      help="Only handle experiments matching all the given pattern. "
                           "If empty, handle all experiments")
    grid.set_defaults(action=grid_action)

    run = subparsers.add_parser("run", help="Run locally the given command.")
    run.add_argument("-f", "--from_sig", help="Signature of job to use as baseline.")
    run.add_argument("-d", "--ddp", action="store_true", help="Distributed training.")
    run.add_argument("argv", nargs='*')
    run.set_defaults(action=run_action)

    launch = subparsers.add_parser("launch")
    launch.add_argument("-f", "--from_sig", help="Signature of job to use as baseline.")
    launch.add_argument("-a", "--attach", action="store_true",
                        help="Attach to the remote process. Interrupting the command will "
                             "kill the remote job.")
    launch.add_argument("--no_tail", action="store_false", dest="tail", default=True,
                        help="Does not tail the log once job is started.")
    launch.add_argument("-C", "--cancel", action='store_true',
                        help="Cancel any existing job and return.")
    add_submit_rules(launch)
    add_slurm_config(launch)
    launch.add_argument("argv", nargs='*')
    launch.set_defaults(action=launch_action)

    info = subparsers.add_parser("info")
    info.add_argument("-f", "--from_sig", help="Signature of job to use as baseline.")
    info.add_argument("-j", "--job_id", help="Find job by job id.")
    info.add_argument("-C", "--cancel", action="store_true", help="Cancel job")
    info.add_argument("-l", "--log", action="store_true", help="Show entire log")
    info.add_argument("-t", "--tail", action="store_true", help="Tail log")
    info.add_argument("-m", "--metrics", action="store_true", help="Show last metrics")
    info.add_argument("argv", nargs='*')
    info.set_defaults(action=info_action)

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    if args.action is None:
        fatal("You must give an action.")

    if args.package is None:
        args.package = _find_package()
    module_name = args.package + ".train"
    sys.path.insert(0, str(Path(".").resolve()))
    module = import_or_fatal(module_name)
    try:
        main = module.main
    except AttributeError:
        fatal(f"Could not find function `main` in {module_name}.")

    if not isinstance(main, DecoratedMain):
        breakpoint()
        fatal(f"{module_name}.main was not decorated with `dora.main`.")

    if getattr(args, 'from_sig', None) is not None:
        try:
            argv = main.get_argv_from_sig(args.from_sig)
        except RuntimeError:
            fatal(f"Could not find an existing run with sig {args.from_sig}")
        simple_log("Parser", "Injecting argv", argv, "from sig", args.from_sig)
        args.argv = argv + args.argv

    args.action(args, main)


if __name__ == "__main__":
    main()
