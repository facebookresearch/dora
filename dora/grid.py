from collections import OrderedDict
from functools import partial
from importlib import import_module
from pathlib import Path
import pkgutil
import pickle
import typing as tp
import sys

from .conf import SlurmConfig, SubmitRules, update_from_args
from .explore import Explorer, Launcher
from .main import DecoratedMain
from .log import simple_log, fatal
from .shep import Sheep, Shepherd, no_log
from .utils import try_load

import treetable as tt

log = partial(simple_log, "Grid:")


def get_explore(args, main):
    package = args.package
    root_name = package + ".grids"
    try:
        grids = import_module(root_name)
    except ImportError:
        fatal(f"Could not find module {root_name}.")

    if args.grid is None:
        candidates = []
        for info in pkgutil.walk_packages([Path(grids.__file__).parent]):
            mod = import_module(root_name + "." + info.name)
            if hasattr(mod, 'explore'):
                candidates.append(info.name)
        log(f"Potential grids are: {', '.join(candidates)}")
        sys.exit(0)

    grid_name = root_name + "." + args.grid
    try:
        grid = import_module(grid_name)
    except ImportError:
        fatal(f"Could not import {grid_name}")

    try:
        explore = grid.explore
    except AttributeError:
        fatal(f"{grid_name} has no exploration function `explore`.")
    if not isinstance(explore, Explorer):
        explore = Explorer(explore, main.custom)
    return explore


def grid_action(args, main: DecoratedMain):
    explore = get_explore(args, main)

    shepherd = Shepherd(main, log=log if args.verbose else no_log)
    slurm = main.custom.get_slurm_config()
    rules = SubmitRules()
    update_from_args(rules, args)

    grid_file = shepherd.grids / (args.grid + ".pkl")
    previous_herd = try_load(grid_file) or {}

    herd = OrderedDict()
    shepherd = Shepherd(main)
    launcher = Launcher(shepherd, herd)
    explore(launcher)

    shepherd.update()

    for sheep, slurm in herd.values():
        shepherd.maybe_submit_lazy(sheep, slurm, rules)
        previous_herd.pop(sheep.run.sig)

    for sheep, _ in previous_herd:
        if not sheep.is_done():
            shepherd.cancel_lazy(sheep)
            name = main.get_name(sheep.run)
            log(f"Canceling job {sheep.job.job_id} for no longer required sheep {name}")

    shepherd.commit()
    pickle.dump(herd, open(grid_file, "wb"))

    monitor(args, main, list(herd.values()))


def monitor(args, main: DecoratedMain, herd: tp.List[Sheep]):
    if not herd:
        log("No experiments")
        return


