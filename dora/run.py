from functools import partial
import os
from shutil import rmtree
import sys
import typing as tp
import time

from . import clean_git
from .executor import start_ddp_workers
from .main import DecoratedMain
from .log import simple_log, red, fatal
from .shep import Shepherd

log = partial(simple_log, "Launch:")


def check_job_and_clear(argv: tp.List[str], main: DecoratedMain, clear: bool = False):
    """This will check if an existing job is running and warn,
    unless --clear is passed, in which case we must cancel it.
    """
    shepherd = Shepherd(main, log)
    sheep = shepherd.get_sheep_from_argv(argv)
    if sheep.job is not None:
        shepherd.update()
        if not sheep.is_done():
            job = sheep.job
            log(red(f"Found existing slurm job {job.job_id} with status {job.state}."))
            if clear:
                log("Cancelling the existing job.")
                shepherd.cancel_lazy(sheep.job)
                shepherd.commit()
                time.sleep(3)
            else:
                log(red("PLEASE ABORT NOW UNLESS YOU ARE SURE OF WHAT YOU DO."))
    if clear and sheep.xp.folder.exists():
        log("Removing existing XP folder.")
        try:
            rmtree(sheep.xp.folder)
        except OSError:
            log("Failed to properly remove folder, but things should be okay...")


def do_clean_git(args, main: DecoratedMain):
    if '_DORA_CLEAN_GIT_DONE' in os.environ:
        return
    if not main.dora.clean_git:
        if args.clean_git:
            fatal("--clean_git can only be used if dora.clean_git is True.")
        return
    if not args.clean_git:
        return

    xp = main.get_xp(args.argv)
    main.init_xp(xp)

    clean_git.check_repo_clean()
    clean_git.shallow_clone(xp.code_folder)

    # Let's move to the right folder
    exec_dir = clean_git.get_clone_exec_dir(xp)
    os.chdir(exec_dir)
    os.environ['_DORA_CLEAN_GIT_DONE'] = '1'
    os.environ['_DORA_DIR_OVERRIDE'] = main.dora.dir
    os.execv(sys.executable, [sys.executable, "-m", "dora"] + sys.argv[1:])


def run_action(args, main: DecoratedMain):
    if args.ddp and not os.environ.get('RANK'):
        check_job_and_clear(args.argv, main, args.clear)
        do_clean_git(args, main)
        start_ddp_workers(args.package, main, args.argv)
    else:
        if 'WORLD_SIZE' not in os.environ:
            check_job_and_clear(args.argv, main, args.clear)
            do_clean_git(args, main)
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
        sys.argv[1:] = args.argv
        main()
