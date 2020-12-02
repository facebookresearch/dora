from collections import namedtuple
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
from omegaconf import OmegaConf


class _NotThere:
    pass


NotThere = _NotThere()
_Difference = namedtuple("_Difference", "path key ref other ref_value other_value")


def _compare_config(ref, other, path=[]):
    """
    Given two configs, gives an iterator over all the differences. For each difference,
    this will give a _Difference namedtuple.
    The value `NotThere` is used when a key is missing on one side.
    """
    keys = sorted(ref.keys())
    remaining = sorted(set(other.keys()) - set(ref.keys()))
    delta = []
    path.append(None)
    for key in keys:
        path[-1] = key
        ref_value = ref[key]
        try:
            other_value = other[key]
        except KeyError:
            other_value = NotThere

        different = False
        if isinstance(ref_value, DictConfig):
            if isinstance(other_value, DictConfig):
                yield from _compare_config(ref_value, other_value, path)
            else:
                different = True
                yield _Difference(list(path), key, ref, other, ref_value, other_value)
        elif other_value != ref_value:
            different = True
        if different:
            yield _Difference(list(path), key, ref, other, ref_value, other_value)
    for key in remaining:
        path[-1] = key
        yield _Difference(list(path), key, ref, other, NotThere, other_value)
    path.pop(-1)
    return delta


def _is_excluded(key, excluded):
    for pattern in excluded:
        if fnmatch(key, pattern):
            return True


class HydraSupport:
    _EXCLUDED = ["dora.*", "slurm.*"]

    def __init__(self, module, config_name, config_path=None):
        self.config_name = config_name
        if module == "__main__":
            spec = sys.modules[module].__spec__
            if spec is None:
                module_path = sys.argv[0]
                self.job_name = module_path.rsplit(".", 2)[1]
            else:
                module_path = spec.origin
                module = spec.name
                self.job_name = module.rsplit(".", 1)[1]
        else:
            module_path = find_spec(module).origin
        self.config_path = Path(module_path).parent.resolve()
        if config_path is not None:
            self.config_path = self.config_path / config_path

    def _get_config(self, overrides=[], return_hydra_config=False):
        with initialize_config_dir(str(self.config_path), job_name=self.job_name):
            return compose(self.config_name, overrides, return_hydra_config=return_hydra_config)

    def get_config(self, overrides=[], return_hydra_config=False):
        """
        Returns the config with the actual signature computation.
        """
        sig = self.get_signature(overrides)
        return self._get_config(overrides + [f"dora.sig={sig}"], return_hydra_config)

    def get_delta(self, init, other):
        delta = _compare_config(init, other)
        excluded = self._EXCLUDED + list(init.dora.exclude)
        delta = []
        for diff in _compare_config(init, other):
            name = ".".join(diff.path)
            if not _is_excluded(name, excluded):
                delta.append((name, diff.other_value))
        delta.sort(key=lambda x: x[0])
        return delta

    def get_signature(self, overrides=[]):
        init = self._get_config()
        other = self._get_config(overrides)
        delta = self.get_delta(init, other)
        return sha1(json.dumps(delta).encode('utf8')).hexdigest()[:16]

    def get_run_dir(self, overrides=[]):
        sig = self.get_signature(overrides)
        cfg = self._get_config(overrides + [f"dora.sig={sig}"], True)
        return cfg.hydra.run.dir

    def get_run_dir_from_sig(self, sig: str):
        cfg = self._get_config([f"dora.sig={sig}"], True)
        return Path(cfg.hydra.run.dir)

    def get_config_from_sig(self, sig: str, return_hydra_config=False):
        path = self.get_run_dir_from_sig(sig)
        if not path.is_dir():
            raise RuntimeError(f"Could not find experiment with signature {sig}")
        cfg = OmegaConf.load(path / ".hydra/config.yaml")
        if return_hydra_config:
            cfg.hydra = OmegaConf.load(path / ".hydra/hydra.yaml").hydra
        return cfg

    def get_overrides_from_sig(self, sig: str):
        cfg = self.get_config_from_sig(sig, return_hydra_config=True)
        path = Path(cfg.hydra.run.dir)
        if not path.is_dir():
            raise RuntimeError(f"Could not find experiment with signature {sig}")
        overrides = OmegaConf.load(path / ".hydra/overrides.yaml")
        excluded = self._EXCLUDED + list(cfg.dora.exclude)
        overrides = [o for o in overrides if not _is_excluded(o.split('=', 1)[0], excluded)]
        return overrides


def main(config_name, config_path=None):

    # A lot of black magic happens here. We are going to intercept the overrides,
    # and compute both the reference config (i.e. without overrides) and the actual one.
    # From the difference between the two configs, we can derive a unique experiment
    # signature. .
    def _decorate(_main):
        @wraps(_main)
        def _decorated():
            support = HydraSupport(_main.__module__, config_name, config_path)
            overrides = [a for a in sys.argv[1:] if not a.startswith('-')]
            sig = support.get_signature(overrides)
            sys.argv.append(f"dora.sig={sig}")
            return hydra.main(
                config_name=config_name,
                config_path=config_path)(_main)()
        _decorated.config_name = config_name
        _decorated.config_path = config_path
        return _decorated
    return _decorate
