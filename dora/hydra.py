from fnmatch import fnmatch
from functools import wraps
from hashlib import sha1
from importlib.util import find_spec
import json
from pathlib import Path
import sys

import hydra
from hydra.experimental import compose, initialize_config_dir
from omegaconf.dictconfig import DictConfig


class _NotThere:
    pass


NotThere = _NotThere()


def _compare_config(init, other, path=[]):
    keys = sorted(init.keys())
    remaining = sorted(set(other.keys()) - set(init.keys()))
    delta = []
    path.append(None)
    for key in keys:
        path[-1] = key
        name = ".".join(path)
        ref = init[key]
        try:
            val = other[key]
        except KeyError:
            val = NotThere
        if isinstance(ref, DictConfig):
            if isinstance(val, DictConfig):
                delta += _compare_config(ref, val, path)
            else:
                delta += [(name, val)]
        elif val != ref:
            delta += [(name, val)]
    for key in remaining:
        path[-1] = key
        name = ".".join(path)
        delta += [(name, other[key])]
    path.pop(-1)
    return delta


def _is_excluded(key, excluded):
    for pattern in excluded:
        if fnmatch(key, pattern):
            return True


class HydraSupport:
    def __init__(self, module, config_name, config_path=None):
        self.config_name = config_name
        if module == "__main__":
            spec = sys.modules[module].__spec__
            if spec is None:
                module_path = sys.argv[0]
            else:
                module_path = spec.origin
        else:
            module_path = find_spec(module).origin
        self.config_path = Path(module_path).parent
        if config_path is not None:
            self.config_path = self.config_path / config_path

    def get_config(self, overrides=[], return_hydra_config=False):
        with initialize_config_dir(str(self.config_path), job_name="dora"):
            return compose(self.config_name, overrides, return_hydra_config=return_hydra_config)

    def get_delta(self, overrides=[]):
        init = self.get_config()
        other = self.get_config(overrides)
        delta = _compare_config(init, other)
        excluded = ["dora.*"] + init.dora.exclude
        delta = [(key, value) for key, value in deltas if not _is_excluded(key, excluded)]
        delta.sort()
        return delta

    def get_signature(self, overrides=[]):
        delta = self.get_delta(overrides)
        return sha1(json.dumps(delta).encode('utf8')).hexdigest()

    def get_run_dir(self, overrides=[]):
        pass




def main(config_name, config_path=None):
    def _decorate(_main):
        @wraps(_main)
        def _decorated():
            support = HydraSupport(_main.__module__, config_name, config_path)
            overrides = [a for a in sys.argv[1:] if not a.startswith('-')]
            sig = support.get_signature(overrides)
            sys.argv.append(f"dora.sig={sig}")
            return hydra.main(config_name=config_name, config_path=config_path)(_main)()
        return _decorated
    return _decorate
