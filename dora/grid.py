"""
This defines the `dora grid` action, and provides a `run_grid` API
that can be used from a notebook or any other script.

When using the API, you can provide the equivalent of the command line flags
with the `RunGridArgs` dataclass.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, field
import fnmatch
from functools import partial
import os
from pathlib import Path
import pkgutil
import typing as tp
import shutil
import sys
import time

from .conf import SlurmConfig, SubmitRules, update_from_args
from .explore import Explorer, Launcher
from .main import DecoratedMain
from .log import colorize, simple_log, fatal
from .shep import Sheep, Shepherd
from .utils import import_or_fatal, try_load

import treetable as tt

log: tp.Callable[[str], None] = partial(simple_log, "Grid:")


@dataclass
class RunGridArgs:
    """
    Arguments to tune the behavior of the `run_grid` function.

    Args:
        patterns (list[str]): List of patterns used to filter by name
            the XPs.
        monitor (bool): if True, will monitor the advances of the XPs
            every `interval` minutes, stopping only when all runs completed or
            failed.
        interval (float): interval in minutes to wait between updates.
        trim (int or None): if provided, will trim all XP logs to the epoch of
            the XP with the provided index. Useful to compare XP started at different
            times.
        trim_last (bool): if True, will trim all XP to the least advanced XP.
        dry_run (bool): if True, Dora will simulate the run of the grid, without scheduling
            or canceling any XP.
        cancel (bool): if True, will cancel all XPs in the grid. If `patterns` is provided,
            only XP matching the patterns will be canceled.
        clear (bool): This will cancel any previous job, clear the XP folder,
            and reschedule a new experiment.

    """
    patterns: tp.List[str] = field(default_factory=list)

    # Monitoring params
    monitor: bool = True
    interval: float = 5
    trim: tp.Optional[int] = None
    trim_last: bool = False

    # Scheduling
    dry_run: bool = False
    cancel: bool = False
    clear: bool = False
    init: tp.Optional[bool] = False

    # jupyter display?
    jupyter: bool = False

    # Other flags, supported only from the command line.
    folder: tp.Optional[int] = None
    log: tp.Optional[int] = None
    tail: tp.Optional[int] = None

    _from_commandline: bool = False


def _get_explore(args, main):
    # Finds the explorer.
    package = args.package
    root_name = package + ".grids"
    grids = import_or_fatal(root_name)

    if args.grid is None:
        candidates = []
        for info in pkgutil.walk_packages([Path(grids.__file__).parent]):
            if not info.name.startswith('_'):
                candidates.append(info.name)
        log(f"Potential grids are: {', '.join(candidates)}")
        sys.exit(0)

    grid_name = root_name + "." + args.grid
    grid = import_or_fatal(grid_name)

    try:
        explorer = grid.explorer
    except AttributeError:
        fatal(f"{grid_name} has no exploration function `explorer`.")
    if not isinstance(explorer, Explorer):
        fatal(f"{explorer} must be an instance of `dora.Explorer`")
    return explorer


def grid_action(args: tp.Any, main: DecoratedMain):
    explorer = _get_explore(args, main)
    slurm = main.get_slurm_config()
    update_from_args(slurm, args)
    rules = SubmitRules()
    update_from_args(rules, args)
    grid_args = RunGridArgs()
    grid_args._from_commandline = True
    update_from_args(grid_args, args)
    run_grid(main, explorer, args.grid, rules, slurm, grid_args)


def run_grid(main: DecoratedMain, explorer: Explorer, grid_name: str,
             rules: SubmitRules = SubmitRules(), slurm: tp.Optional[SlurmConfig] = None,
             args: RunGridArgs = RunGridArgs()) -> tp.List[Sheep]:
    """
    Run a grid search, this is the API underlying the `dora grid` command,
    so that it can be used from a notebook.
    You can also provide patterns to filter out XPs to be displayed.

    Args:
        main (DecoratedMain): main training function, decorated with Dora.
        explorer (Explorer): explorer instance that will define the XPs to launch.
        grid_name (str): this must be a unique name for the grid.
        rules (SubmitRules): see `dora.conf.SubmitRules`, those defines the
            rules for rescheduling failed XP etc.
        slurm (SlurmConfig or None): if provided, this will override
            the default Slurm config defined my the `main` argument.

    Returns:
        A list of `dora.shep.Sheep`.

    """
    assert isinstance(explorer, Explorer)
    if slurm is None:
        slurm = main.get_slurm_config()

    grid_folder = main.dora.dir / main.dora.grids / grid_name
    grid_folder.mkdir(exist_ok=True, parents=True)

    herd: OrderedDict[str, tp.Tuple[Sheep, SlurmConfig]] = OrderedDict()
    shepherd = Shepherd(main, log=log)
    if main._slow:
        pending: OrderedDict[int, Future] = OrderedDict()
        with ProcessPoolExecutor(4) as pool:
            launcher = Launcher(shepherd, slurm, pending, pool=pool)
            explorer(launcher)
            for future in pending.values():
                sheep, slurm = future.result()
                assert isinstance(sheep, Sheep) and isinstance(slurm, SlurmConfig)
                assert sheep.xp.sig is not None
                herd[sheep.xp.sig] = (sheep, slurm)
    else:
        launcher = Launcher(shepherd, slurm, herd)
        explorer(launcher)

    shepherd.update()
    sheeps = [sheep for sheep, _ in herd.values()]
    sheeps = _filter_grid_sheeps(args.patterns, main, sheeps)

    if args.clear:
        if args.dry_run:
            fatal("--dry_run is incompatible with --clear.")
        log(f"You are about to restart {len(sheeps)} experiments from the grid {grid_name} "
            "from scratch. This cannot be reverted.")
        if args._from_commandline:
            repl = input("Confirm [yN]: ")
            if repl.lower() != "y":
                fatal("Abort...")
        log("Canceling all current jobs...")
        for sheep in sheeps:
            if sheep.job is not None:
                shepherd.cancel_lazy(sheep.job)
        shepherd.commit()
        log("Deleting XP folders...")
        for sheep in sheeps:
            if sheep.xp.folder.exists():
                shutil.rmtree(sheep.xp.folder)
            sheep.job = None

    if not args.cancel:
        for sheep in sheeps:
            _, slurm = herd[sheep.xp.sig]
            shepherd.maybe_submit_lazy(sheep, slurm, rules)

    to_unlink = []
    old_sheeps = []
    for child in grid_folder.iterdir():
        if child.name not in herd:
            to_unlink.append(child)
            try:
                old_sheep = shepherd.get_sheep_from_sig(child.name)
            except Exception as error:
                log(f"Error when trying to load old sheep {child.name}: {error}")
                # We fallback on manually loading the job file.
                job_file = child / main.dora.shep.job_file
                job = try_load(job_file)
                if job is not None:
                    log(f"Canceling job {job.job_id} from unloadable sheep {child.name}.")
                    shepherd.cancel_lazy(job)
            else:
                assert old_sheep is not None
                old_sheeps.append(old_sheep)

    shepherd.update()
    for old_sheep in old_sheeps:
        if not old_sheep.is_done():
            assert old_sheep.job is not None
            shepherd.cancel_lazy(old_sheep.job)
            name = main.get_name(old_sheep.xp)
            log(f"Canceling job {old_sheep.job.job_id} for no longer required "
                f"sheep {old_sheep.xp.sig}/{name}")

    if args.cancel:
        for sheep in sheeps:
            if not sheep.is_done():
                assert sheep.job is not None
                name = main.get_name(sheep.xp)
                log(f"Canceling job {sheep.job.job_id} for sheep {sheep.xp.sig}/{name}")
                shepherd.cancel_lazy(sheep.job)

    if not args.dry_run:
        for sheep in sheeps:
            link = (grid_folder / sheep.xp.sig)
            if link.exists() or link.is_symlink():
                assert link.is_symlink() and link.resolve() == sheep.xp.folder
            else:
                link.symlink_to(sheep.xp.folder)
        shepherd.commit()

        for child in to_unlink:
            child.unlink()
    if args.init:
        for sheep in sheeps:
            main.init_xp(sheep.xp)

    if args.cancel:
        return sheeps

    if not sheeps:
        log("No sheep to handle.")
        return sheeps

    actions = [action for action in [args.folder, args.log, args.tail] if action is not None]

    if actions:
        if not args._from_commandline:
            raise RuntimeError("The folder, log, and tail "
                               "flags are only supported from the command line.")
        assert len(actions) == 1
        index = actions[0]
        try:
            sheep = sheeps[index]
        except IndexError:
            fatal(f"Invalid index {args.folder}")
        name = main.get_name(sheep.xp)
        if args.folder is not None:
            print(sheep.xp.folder)
        elif args.tail is not None:
            if not sheep.log.exists():
                fatal(f"Log {sheep.log} does not exist")
            os.execvp("tail", ["tail", "-n", "200", "-f", sheep.log])
        else:
            if not sheep.log.exists():
                fatal(f"Log file does not exist for sheep {name}.")
            try:
                shutil.copyfileobj(open(sheep.log), sys.stdout)
            except BrokenPipeError:
                pass
        return sheeps

    print(f"Monitoring Grid {grid_name}")
    while True:
        if args.jupyter:
            from IPython import display
            display.clear_output(wait=True)
        shepherd.update()
        if monitor(args, main, explorer, sheeps):
            # All jobs finished or failed, stop monitoring
            break
        if not args.monitor:
            break
        sleep = int(args.interval * 60)
        print()
        for ela in range(sleep):
            out = f'Next update in {sleep - ela:.0f} seconds       '
            if sleep - ela < 10:
                out = colorize(out, '31')
            print(out, end='\r')
            time.sleep(1)
        print(' ' * 60)
    return sheeps


def _match_name(name, patterns):
    if not patterns:
        return True
    for pattern in patterns:
        neg = False
        if pattern[:1] == '!':
            pattern = pattern[1:]
            neg = True
        result = fnmatch.fnmatch(name, '*' + pattern + '*')
        if neg:
            if result:
                return False
        elif not result:
            return False
    return True


def _filter_grid_sheeps(patterns: tp.List[str], main: DecoratedMain,
                        sheeps: tp.List[Sheep]) -> tp.List[Sheep]:
    indexes = []
    for p in list(patterns):
        try:
            indexes.append(int(p))
        except ValueError:
            continue
        else:
            patterns.remove(p)
    out = []
    for sheep in sheeps:
        name = main.get_name(sheep.xp)
        if _match_name(name, patterns):
            out.append(sheep)
    if indexes:
        out = [out[idx] for idx in indexes]
    return out


def monitor(args: tp.Any, main: DecoratedMain, explorer: Explorer, herd: tp.List[Sheep]) -> bool:
    """Single iteration of monitoring of the jobs in a Grid.
    Returns `True` if all jobs are done or failed, and `False` otherwise.
    """
    names, base_name = main.get_names([sheep.xp for sheep in herd])
    histories = [main.get_xp_history(sheep.xp) for sheep in herd]

    trim = None
    if args.trim is not None:
        trim = len(histories[args.trim])
    elif args.trim_last:
        trim = min(len(metrics) for metrics in histories)

    if trim is not None:
        histories = [metrics[:trim] for metrics in histories]

    lines = []
    finished = True
    for index, (sheep, history, name) in enumerate(zip(herd, histories, names)):
        state = sheep.state()
        if not sheep.is_done():
            finished = False
        if state is None:
            state = "N/A"
        else:
            state = state[:3]
        meta = {
            'name': name,
            'index': index,
            'sid': sheep.job.job_id if sheep.job else '',
            'sig': sheep.xp.sig,
            'state': state,
        }
        line = {}
        line['Meta'] = meta
        line.update(explorer.process_history(history))
        lines.append(line)

    if base_name:
        print("Base name: ", base_name)
    table = tt.table(
        shorten=True,
        groups=[
            tt.group("Meta", explorer.get_grid_meta()),
        ] + explorer.get_grid_metrics()
    )
    print(tt.treetable(lines, table, colors=explorer.get_colors()))
    return finished
