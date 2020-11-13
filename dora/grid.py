# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Alexandre Défossez, 2020
"""
Dora the Explorer, special thank to @pierrestock.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import getpass
import json
from pathlib import Path
import os
import shlex
import shutil
import subprocess as sp
import sys
import time

import treetable as tt
from treetable.text import colorize
import tqdm


def fatal(first, *args):
    # Print first in bold
    print(colorize(first, '1'), *args, file=sys.stderr)
    sys.exit(1)


def ask_train(args, parse=True):
    cmd = ['./train.py'] + args
    proc = sp.run(cmd, capture_output=True)
    cmd = " ".join(shlex.quote(c) for c in cmd)
    stdout = proc.stdout.decode('utf-8')
    stderr = proc.stderr.decode('utf-8')
    if proc.returncode:
        fatal("Error calling ./train.py:", cmd, "\n", stderr)
    if not parse:
        return
    try:
        return json.loads(stdout)
    except ValueError:
        fatal("Error parsing json from command", cmd, stdout, "\n", stderr)


def match_name(name, patterns):
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


def shorten(names):
    names = [name_for_human(name) for name in names]
    commons = set(names[0].split(" "))
    for name in names:
        commons &= set(name.split(" "))
    out = []
    for name in names:
        name = " ".join(p for p in name.split(" ") if p not in commons)
        out.append(name)
    return " ".join(commons), out


def registerer(base_args, pool):
    registry = []

    def register(*args):
        all_args = dict(base_args)
        for arg in args:
            all_args.update(arg)
        all_args = [f"{k}={v}" for k, v in all_args.items()]
        registry.append((all_args, pool.submit(ask_train, all_args + ["name=1"])))
        return registry[-1][1]
    return registry, register


def load_registry(path):
    try:
        return json.load(open(path, "rb"))
    except IOError:
        return {}


def save_registry(registry, path):
    tmp = str(path) + ".tmp"
    json.dump(registry, open(tmp, "w"))
    os.rename(tmp, path)


def garbage_collect(olds, news):
    canceled = []
    for old in dict(olds):
        # old exp is not used anymore, cancel it (noop if not running or already done)
        if old not in news:
            ask_train(olds.pop(old) + ["cancel=1"], parse=False)
            canceled.append(old)
    return canceled


def name_for_human(name):
    prefix = "exp_"
    name = name[len(prefix):]
    parts = name.split(",")
    out = []
    for part in parts:
        k, v = part.split('=', 1)
        *ks, last = k.split('.')
        if '_' in last:
            last = '_'.join(p[:3] for p in last.split('_'))

        k = '.'.join([p[:3] for p in ks] + [last])
        out.append(f'{k}={v}')
    return ' '.join(out)


def update(registry, submit=False, cfg=None):
    test_metrics = ['sisnr', 'stoi', 'pesq']
    lines = []
    extra = [
        'query=1',
        'tail=0',
        f'submit={submit}',
    ]
    if submit:
        extra += [
            f'retry={cfg.retry}',
            f'replace_done={cfg.replace_done}',
            f'replace={cfg.replace}',
            f'replace_pending={cfg.update}',
        ]
    outs = []
    items = list(registry.items())
    with ThreadPoolExecutor(4) as pool:
        for _, args in items:
            outs.append(pool.submit(ask_train, args + extra))
        outs = [out.result() for out in tqdm.tqdm(outs, leave=False, ncols=120)]

    if cfg.trim is not None:
        length = len(outs[cfg.trim]['history'] or [])
        for out in outs:
            if out['history']:
                out['history'] = out['history'][:length]
    common, names = shorten(registry.keys())
    print("Common config:", common)
    for index, name in enumerate(names):
        line = {}
        meta = {'name': name, 'index': index}

        out = outs[index]
        meta['sid'] = out['sid'] or ''
        meta['state'] = out['state'][:3]
        line['meta'] = meta

        history = out['history'] or []
        epoch = len(history)
        train = {'epoch': epoch}
        test = {}
        if history:
            train.update({
                'train': 100 * history[-1]['train'],
                'valid': 100 * history[-1]['valid'],
                'best': 100 * history[-1]['best'],
            })
        for metrics in history:
            for name in test_metrics:
                if name in metrics:
                    test[name] = metrics[name]
        line['train'] = train
        line['test'] = test
        lines.append(line)

    table = tt.table(
        shorten=True,
        groups=[
            tt.group("meta", [
                tt.leaf("index", align=">"),
                tt.leaf("name"),
                tt.leaf("state"),
                tt.leaf("sid", align=">"),
            ]),
            tt.group("train", [
                tt.leaf("epoch"),
                tt.leaf("train", ".3f"),
                tt.leaf("valid", ".3f"),
                tt.leaf("best", ".3f"),
             ], align=">"),
            tt.group("test", [
                tt.leaf(name, ".3f")
                for name in test_metrics
             ], align=">")
        ]
    )
    print(tt.treetable(lines, table, colors=["0", "38;5;245"]))


def collect(registry):
    import torch
    _, names = shorten(registry.keys())
    fullnames = registry.keys()
    out = Path.home() / "denoising/samples"
    models = Path.home() / "denoising/models"
    if out.exists():
        shutil.rmtree(out)
    if models.exists():
        shutil.rmtree(models)
    out.mkdir(parents=True)
    models.mkdir(parents=True)
    for idx, (name, fullname, args) in enumerate(zip(names, fullnames, registry.values())):
        folder = ask_train(args + ['name=1'])['folder']
        src = Path(folder) / "samples"
        if not src.exists():
            continue
        if not name:
            name = "default"
        dst = out / name
        dst.mkdir()
        print("Exporting", "'" + name + "'")
        for file in src.iterdir():
            if file.suffix == ".wav":
                proc = sp.run(['lame', '-b320', file, dst / (file.stem + ".mp3")],
                              stdout=sp.DEVNULL, stderr=sp.DEVNULL)
                if proc.returncode:
                    fatal("lame crashed converting", file)
        cp = torch.load(Path(folder) / 'checkpoint.th', 'cpu')
        cp['model']['state'] = cp.pop('best_state')
        cp['name'] = fullname
        del cp['optimizer']
        model_path = models / (name + '.th')
        torch.save(cp, model_path)
        link = models / str(idx)
        link.symlink_to(model_path)
    return


def main(explorer, name, base_args=None):
    parser = argparse.ArgumentParser("dora.py")
    default_db = Path("/checkpoint/" + getpass.getuser() + "/denoising/dora/")
    parser.add_argument("-r", "--retry", action="store_true",
                        help="Retry failed jobs")
    parser.add_argument("-R", "--replace", action="store_true",
                        help="Replace any running job.")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from 0 any unfinished job.")
    parser.add_argument("-D", "--replace_done", action="store_true",
                        help="Also resubmit done jobs.")
    parser.add_argument("-U", "--update", action="store_true",
                        help="Only replace jobs that are still pending.")
    parser.add_argument("-C", "--cancel", action='store_true',
                        help="Cancel all running jobs.")
    parser.add_argument("--db", default=default_db, type=Path,
                        help="Where to store info across runs of dora.")
    parser.add_argument("-i", "--interval", default=5, type=float,
                        help="Update status and metrics every that number of minutes. "
                             "Default is 5 min.")

    # commands to learn more about some jobs
    parser.add_argument("-f", "--folder", type=int,
                        help="Show the folder for the job with the given index")
    parser.add_argument("-l", "--log", type=int,
                        help="Show the log for the job with the given index")
    parser.add_argument("--collect", action="store_true",
                        help="Collect name files, in ~/denoising/samples")

    parser.add_argument("-t", "--trim", type=int,
                        help="Trim history to the length of the exp with the given index.")
    parser.add_argument("patterns", nargs='*',
                        help="Only handle experiments matching all the given pattern. "
                             "If empty, handle all experiments")
    cfg = parser.parse_args()
    cfg.db.mkdir(exist_ok=True, parents=True)
    db = cfg.db / (name + ".db.json")
    olds = load_registry(db)

    with ThreadPoolExecutor(4) as pool:
        registry, register = registerer(base_args or {}, pool)
        # ask explorer to register all experiments it wants
        explorer(register)
        registry = {
            out.result()['name']: args
            for args, out in tqdm.tqdm(registry, leave=False, ncols=120)}
    # Garbage collect unwanted experiments, modifies olds  in place
    for name in garbage_collect(olds, registry):
        print("Canceled", name)
    olds.update(registry)
    # Save registry now, so we don't have to worry after.
    save_registry(olds, db)

    # Now keep only the exps matching the patterns
    patterns = cfg.patterns
    indexes = []
    for p in list(patterns):
        try:
            indexes.append(int(p))
        except ValueError:
            continue
        else:
            patterns.remove(p)
    for name in dict(registry):
        if not match_name(name_for_human(name), cfg.patterns):
            registry.pop(name)
    if indexes:
        names = list(registry)
        registry = {names[idx]: registry[names[idx]] for idx in indexes}

    if cfg.cancel:
        for name, args in registry.items():
            print("Canceling", name)
            ask_train(args + ["cancel=1"], parse=False)
        return

    if cfg.folder is not None:
        args = list(registry.values())[cfg.folder] + ["name=1"]
        print(ask_train(args)['folder'])
        return
    if cfg.log is not None:
        cmd = ['./train.py']
        cmd += list(registry.values())[cfg.log] + ["check=1"]
        try:
            sp.run(cmd)
        except KeyboardInterrupt:
            pass
        return
    if cfg.collect:
        collect(registry)
        return

    if cfg.restart:
        ok = input("About to restart experiments from zero [yN]")
        if ok.lower() != "y":
            fatal("Aborting")
        for args in registry.values():
            args += ["restart=1"]

    # First update will submit everything
    update(registry, submit=True, cfg=cfg)
    # Now run in a loop
    try:
        while True:
            sleep = int(cfg.interval * 60)
            for ela in range(sleep):
                print(f'Next update in {sleep - ela:.0f} seconds       ', end='\r')
                time.sleep(1)
            update(registry, cfg=cfg)
    except KeyboardInterrupt:
        return