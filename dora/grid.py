from functools import partial
from importlib import import_module
from pathlib import Path
import pkgutil

from .log import simple_log, fatal

log = partial(simple_log, "Grid:")


class Scheduler:
    def __init__(self, main):
        self.main = main

    def __call__(self, *args):
        pass


def grid_action(args, main):
    package = args.package

    root_name = package + ".grids"
    if args.grid is None:
        try:
            grids = import_module(root_name)
        except ImportError:
            fatal(f"Could not find module {root_name}.")
        candidates = []
        for info in pkgutil.walk_packages([Path(grids.__file__).parent]):
            mod = import_module(root_name + "." + info.name)
            if hasattr(mod, 'explore'):
                candidates.append(info.name)
        log(f"Potential grids are: {', '.join(candidates)}")
        return

    grid_name = root_name + "." + args.grid
    try:
        grid = import_module(grid_name)
    except ImportError:
        fatal(f"Could not import {grid_name}")

    try:
        explore = grid.explore
    except AttributeError:
        fatal(f"{grid_name} has no exploration function `explore`.")

    explore(scheduler)
