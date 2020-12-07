"""
This module provides support for Hydra, in particular the `main` wrapper between
the end user `main` function and Hydra.
"""
from collections import namedtuple
from fnmatch import fnmatch
from hashlib import sha1
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import typing as tp

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


def overrides_from_argv(argv):
    return [a for a in argv if not a.startswith('-')]


class DecoratedMain:
    """
    The core concept in Dora is that of a run *signature*.
    Given a set of Hydra overrides, this class will compare all the fields
    that are impacted in the configuration tree structure. Those differences will be hashed.

    The signature will thus uniquely define a run, and be shared even for complex override
    patterns that would end up being equivalent (i.e. composition vs. manual overrides).

    The run dir will be uniquely defined by the signature. If you run twice the same job
    with equivalent parameters, they will run in the same folder, with the end goal of
    automatically discovering checkpoints from a previous run for instance.

    It is even possible to recover a configuration or list of overrides from the signature,
    but only if a previous run used that specific signature.
    """
    _EXCLUDED = ["dora.*", "slurm.*"]

    def __init__(self, main: tp.Callable[[], None], config_name: str, config_path: str):
        self.main = main
        self.config_name = config_name
        self.config_path = config_path
        module = main.__module__
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
            self.job_name = module.rsplit(".", 1)[1]
        self.full_config_path = Path(module_path).parent.resolve()
        if config_path is not None:
            self.full_config_path = self.config_path / config_path

    def __call__(self):
        overrides = overrides_from_argv(sys.argv[1:])
        sig = self.get_signature(overrides)
        sys.argv.append(f"dora.sig={sig}")
        return hydra.main(
            config_name=self.config_name,
            config_path=self.config_path)(self.main)()

    def _get_config(self,
                    overrides: tp.Sequence[str] = [],
                    return_hydra_config: bool = False) -> DictConfig:
        """
        Internal method, returns the config for the given override,
        but without the dora.sig field filled.
        """
        with initialize_config_dir(str(self.full_config_path), job_name=self.job_name):
            return compose(self.config_name, overrides, return_hydra_config=return_hydra_config)

    def get_config(self,
                   overrides: tp.Sequence[str] = [],
                   return_hydra_config: bool = False) -> DictConfig:
        """
        Returns the config for the given override.
        """
        sig = self.get_signature(overrides)
        return self._get_config(overrides + [f"dora.sig={sig}"], return_hydra_config)

    def get_delta(self, init: DictConfig, other: DictConfig):
        """
        Returns an iterator over all the differences between the init and other config.
        """
        delta = _compare_config(init, other)
        excluded = self._EXCLUDED + list(init.dora.exclude)
        delta = []
        for diff in _compare_config(init, other):
            name = ".".join(diff.path)
            if not _is_excluded(name, excluded):
                delta.append((name, diff.other_value))
        delta.sort(key=lambda x: x[0])
        return delta

    def get_signature(self, overrides: tp.Sequence[str] = []) -> str:
        """
        Returns the job signature.
        """
        init = self._get_config()
        other = self._get_config(overrides)
        delta = self.get_delta(init, other)
        return sha1(json.dumps(delta).encode('utf8')).hexdigest()[:16]

    def get_run_dir(self, overrides: tp.Sequence[str] = []) -> Path:
        """
        Get the job run dir.
        """
        sig = self.get_signature(overrides)
        cfg = self._get_config(overrides + [f"dora.sig={sig}"], True)
        return Path(cfg.hydra.run.dir)

    def get_run_dir_from_sig(self, sig: str) -> Path:
        """
        Get the job run dir from the signature.
        """
        cfg = self._get_config([f"dora.sig={sig}"], True)
        return Path(cfg.hydra.run.dir)

    def get_config_from_sig(self, sig: str, return_hydra_config: bool = False) -> DictConfig:
        """
        Get the config from the signature, if a previous run already
        matched that config.
        """
        path = self.get_run_dir_from_sig(sig)
        if not path.is_dir():
            raise RuntimeError(f"Could not find experiment with signature {sig}")
        cfg = OmegaConf.load(path / ".hydra/config.yaml")
        if return_hydra_config:
            cfg.hydra = OmegaConf.load(path / ".hydra/hydra.yaml").hydra
        return cfg

    def get_overrides_from_sig(self, sig: str, exclude: bool = True):
        """
        Get the overrides from the signature, if a previous run matched that config.
        If exclude is True, ignores overrides that would have been excluded in the
        computation of the signature.
        """
        cfg = self.get_config_from_sig(sig, return_hydra_config=True)
        path = Path(cfg.hydra.run.dir)
        if not path.is_dir():
            raise RuntimeError(f"Could not find experiment with signature {sig}")
        overrides = OmegaConf.load(path / ".hydra/overrides.yaml")
        if exclude:
            excluded = self._EXCLUDED + list(cfg.dora.exclude)
            overrides = [o for o in overrides if not _is_excluded(o.split('=', 1)[0], excluded)]
        return overrides


def main(config_name, config_path=None):

    # A lot of black magic happens here. We are going to intercept the overrides,
    # and compute both the reference config (i.e. without overrides) and the actual one.
    # From the difference between the two configs, we can derive a unique experiment
    # signature. .
    def _decorate(_main):
        return DecoratedMain(_main, config_name, config_path)

    return _decorate
