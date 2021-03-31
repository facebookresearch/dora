from collections import OrderedDict
import fnmatch
from functools import partial
from importlib import import_module
from pathlib import Path
import pkgutil
import pickle
import typing as tp
import shutil
import sys
import time

from .conf import SubmitRules, update_from_args
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
        fatal(f"Function {explore} should be decorated with `dora.explorer`")
    return explore


def grid_action(args: tp.Any, main: DecoratedMain):
    explore = get_explore(args, main)

    shepherd = Shepherd(main, log=log if args.verbose else no_log)
    slurm = main.get_slurm_config()
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
        previous_herd.pop(sheep.xp.sig)

    for sheep, _ in previous_herd:
        if not sheep.is_done():
            shepherd.cancel_lazy(sheep)
            name = main.get_name(sheep.xp)
            log(f"Canceling job {sheep.job.job_id} for no longer required sheep {name}")

    shepherd.commit()
    pickle.dump(herd, open(grid_file, "wb"))

    sheeps = list(herd.values())
    sheeps = filter_grid_sheeps(args, main, sheeps)

    if not sheeps:
        log("No sheep to handle.")
        return

    if args.cancel:
        for sheep in sheeps:
            shepherd.cancel_lazy(sheep)
            name = main.get_name(sheep.xp)
            log(f"Canceling job {sheep.job.job_id} for sheep {name}")
        return

    if args.folder is not None or args.log is not None:
        index = args.folder if args.folder is not None else args.log
        try:
            sheep = sheeps[index]
        except IndexError:
            fatal(f"Invalid index {args.folder}")
        name = main.get_name(sheep.xp)
        if args.folder is not None:
            log(f"Folder for sheep {name}")
            print(sheep.xp.folder)
        else:
            if not sheep.xp.log.exists():
                fatal(f"Log file does not exist for sheep {name}.")
            shutil.copyfileobj(open(sheep.xp.log), sys.stdout)
        return

    print(f"Monitoring Grid {args.grid}")
    while True:
        shepherd.update()
        monitor(args, main, sheeps)
        time.sleep(args.interval)


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


def filter_grid_sheeps(args: tp.Any, main: DecoratedMain, sheeps: tp.List[Sheep]) -> tp.List[Sheep]:
    patterns = args.patterns
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
        name = main.get_name(sheep)
        if _match_name(name, patterns):
            out.append(name)
    if indexes:
        return [out[idx] for idx in indexes]


def monitor(args, explorer: Explorer, main: DecoratedMain, herd: tp.List[Sheep]):
    names, base_name = main.get_names([sheep.xp for sheep in herd])
    all_metrics = [main.get_xp_metrics(sheep.xp) for sheep in herd]

    if args.trim is not None:
        length = len(all_metrics[cfg.trim])
        all_metrics = [metrics[:length] for metrics in all_metrics]

        line = []
    for index, (sheep, metrics, name) in enumerate(zip(herd, all_metrics, names)):
        line = {}
        meta = {'name': name, 'index': index}

        meta['sid'] = sheep.job.job_id if sheep.job else ''
        meta['name'] = name
        meta['state'] = sheep.state
        meta['epoch'] = len(metrics)
        line['meta'] = meta
        if metrics:
            line['metrics'] = metrics[-1]
        else:
            line['metrics'] = {}

        lines.append(line)

    print("Base name: ", base_name)
    table = tt.table(
        shorten=True,
        groups=[
            tt.group("meta", [
                tt.leaf("index", align=">"),
                tt.leaf("name"),
                tt.leaf("state"),
                tt.leaf("sid", align=">"),
                tt.leaf("epoch"),
            ]),
            explorer.get_grid_metrics(),
        ]
    )
    print(tt.treetable(lines, table, colors=explorer.get_colors()))
