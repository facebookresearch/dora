# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides support for Hydra, in particular the `main` wrapper between
the end user `main` function and Hydra.
"""
import copy
from collections import namedtuple, OrderedDict
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import sys
import typing as tp
from unittest import mock

import hydra
from hydra.core.global_hydra import GlobalHydra
try:
    from hydra import compose, initialize_config_dir  # type: ignore
except ImportError:
    from hydra.experimental import compose, initialize_config_dir  # type: ignore
    old_hydra = True
else:
    old_hydra = False

from omegaconf.dictconfig import DictConfig

from .conf import DoraConfig, SlurmConfig, update_from_hydra
from .main import DecoratedMain, MainFun
from .xp import XP, get_xp, is_xp

logger = logging.getLogger(__name__)


def _no_copy(self: tp.Any, memo: tp.Any):
    # Dirty trick to speed up Hydra, will remove when Hydra 1.1
    # is released, which solves the issues.
    return self


_Difference = namedtuple("_Difference", "path key ref other ref_value other_value")


class _NotThere:
    pass


NotThere = _NotThere()


def _compare_config(ref, other, path=[]):
    """
    Given two configs, gives an iterator over all the differences. For each difference,
    this will give a _Difference namedtuple.
    """
    keys = sorted(ref.keys())
    remaining = sorted(set(other.keys()) - set(ref.keys()))
    delta = []
    path.append(None)
    for key in keys:
        path[-1] = key
        ref_value = ref[key]
        assert key in other, f"XP config shouldn't be missing any key. Missing key {key}"
        other_value = other[key]

        if isinstance(ref_value, DictConfig):
            assert isinstance(other_value, DictConfig), \
                "Structure of config should be identical between XPs. "\
                f"Wrong type for {key}, expected DictConfig, got {type(other_value)}."
            yield from _compare_config(ref_value, other_value, path)
        elif other_value != ref_value:
            yield _Difference(list(path), key, ref, other, ref_value, other_value)

    for key in remaining:
        path[-1] = key
        other_value = other[key]
        yield _Difference(list(path), key, ref, other, NotThere, other_value)
    path.pop(-1)
    return delta


def _simplify_argv(argv: tp.Sequence[str]) -> tp.List[str]:
    simplified = []
    seen = set()
    for arg in list(argv)[::-1]:
        assert '=' in arg, f'Argument {arg} does not contain ='
        key, value = arg.split('=', 1)
        key = key.strip()
        if key in seen:
            continue
        else:
            seen.add(key)
            simplified.append(arg)
    return simplified[::-1]


def _dump_key(key):
    if key is None:
        return "null"
    elif isinstance(key, (bool, int, float)):
        return str(key)
    elif isinstance(key, str):
        assert ":" not in key
        return key
    else:
        raise TypeError(f"Unsupported dict key type {type(key)} for key {key}")


def _hydra_value_as_override(value):
    # hydra doesn't support parsing dict with the json format, so for now
    # we have to use a custom function to dump a value.
    if value is None:
        return "null"
    elif isinstance(value, (bool, int, float, str)):
        return json.dumps(value)
    elif isinstance(value, dict):
        return "{" + ", ".join(
            f"{_dump_key(key)}: {_hydra_value_as_override(val)}"
            for key, val in value.items()
        ) + "}"
    elif isinstance(value, (list, tuple)):
        return "[" + ", ".join(_hydra_value_as_override(val) for val in value) + "]"
    else:
        raise TypeError(f"Unsupported value type {type(value)} for value {value}")


class HydraMain(DecoratedMain):
    _slow = True

    def __init__(self, main: MainFun, config_name: str, config_path: str, **kwargs):
        self.config_name = config_name
        self.config_path = config_path
        self.hydra_kwargs = kwargs

        module = main.__module__
        if module == "__main__":
            spec = sys.modules[module].__spec__
            if spec is None:
                module_path = sys.argv[0]
                self._job_name = module_path.rsplit(".", 2)[1]
            else:
                assert spec.origin is not None
                module_path = spec.origin
                module = spec.name
                self._job_name = module.rsplit(".", 1)[1]
        else:
            spec = find_spec(module)
            assert spec is not None and spec.origin is not None
            module_path = spec.origin
            self._job_name = module.rsplit(".", 1)[1]
        self.full_config_path = Path(module_path).parent.resolve()
        if config_path is not None:
            self.full_config_path = self.full_config_path / config_path

        self._initialized = False
        self._base_cfg = self._get_config()
        self._config_groups = self._get_config_groups()
        dora = self._get_dora()
        super().__init__(main, dora)
        # this is a really dirty hack to make Hydra believe that this is
        # coming from the __main__ module, as it would usually be.
        # This allows to use relative paths for config_path.
        main.__module__ = "__main__"

    def _get_dora(self) -> DoraConfig:
        dora = DoraConfig()
        if hasattr(self._base_cfg, "dora"):
            update_from_hydra(dora, self._base_cfg.dora)
        dora.exclude += ["dora.*", "slurm.*"]
        dora.dir = Path(dora.dir)
        return dora

    def get_slurm_config(self) -> SlurmConfig:
        """Return default Slurm config for the launch and grid actions.
        """
        slurm = SlurmConfig()
        if hasattr(self._base_cfg, "slurm"):
            update_from_hydra(slurm, self._base_cfg.slurm)
        return slurm

    def get_xp(self, argv: tp.Sequence[str]):
        argv = _simplify_argv(argv)
        cfg = self._get_config(argv)
        base, delta = self._get_base_config(argv)
        delta += self._get_delta(base, cfg)
        xp = XP(dora=self.dora, cfg=cfg, argv=argv, delta=delta)
        return xp

    def value_to_argv(self, arg: tp.Any) -> tp.List[str]:
        # Here we get the raw stuff from what is passed to the grid launcher.
        # arg is either a str (in which case it is a raw override)
        # or a dict, in which case each entry is an override,
        # or a list of dict or a list of str.
        argv = []
        if isinstance(arg, str):
            argv.append(arg)
        elif isinstance(arg, dict):
            for key, value in arg.items():
                if key not in self._config_groups:
                    # We need to convert the value using a custom function
                    # to respect how Hydra parses overrides.
                    value = _hydra_value_as_override(value)
                argv.append(f"{key}={value}")
        elif isinstance(arg, (list, tuple)):
            for part in arg:
                argv += self.value_to_argv(part)
        else:
            raise ValueError(f"Can only process dict, tuple, lists and str, but got {arg}")
        return argv

    def get_name_parts(self, xp: XP) -> OrderedDict:
        parts = OrderedDict()
        assert xp.delta is not None
        for name, value in xp.delta:
            parts[name] = value
        return parts

    def _main(self):
        if is_xp():
            run_dir = f"hydra.run.dir={get_xp().folder}"
            sys.argv.append(run_dir)
        try:
            return hydra.main(
                config_name=self.config_name,
                config_path=self.config_path,
                **self.hydra_kwargs)(self.main)()
        finally:
            if is_xp():
                sys.argv.remove(run_dir)

    def _get_config_groups(self) -> tp.List[str]:
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name,
                                   **self.hydra_kwargs):
            gh = GlobalHydra.instance().hydra
            assert gh is not None
            return list(gh.list_all_config_groups())

    def _is_active(self, argv: tp.List[str]) -> bool:
        if '-m' in argv or '--multirun' in argv:
            return False
        return True

    def _get_base_config(
            self, overrides: tp.List[str] = []
            ) -> tp.Tuple[DictConfig, tp.List[tp.Tuple[str, str]]]:
        """
        Return base config based on composition, along with delta for the
        composition overrides.
        """
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name,
                                   **self.hydra_kwargs):
            gh = GlobalHydra.instance().hydra
            assert gh is not None
            to_keep = []
            delta: tp.List[tp.Tuple[str, str]] = []
            for arg in overrides:
                for group in self._config_groups:
                    if arg.startswith(f'{group}='):
                        to_keep.append(arg)
                        _, value = arg.split('=', 1)
                        delta = [(g, v) for g, v in delta if g != group]
                        delta.append((group, value))
            if not to_keep:
                return self._base_cfg, []
            cfg = self._get_config_noinit(to_keep)
            return cfg, delta

    def _get_config(self,
                    overrides: tp.List[str] = []) -> DictConfig:
        """
        Internal method, returns the config for the given override,
        but without the dora.sig field filled.
        """
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name,
                                   **self.hydra_kwargs):
            return self._get_config_noinit(overrides)

    def _get_config_noinit(self, overrides: tp.List[str] = []) -> DictConfig:
        if old_hydra:
            with mock.patch.object(DictConfig, "__deepcopy__", _no_copy):
                cfg = compose(self.config_name, overrides)  # type: ignore
            cfg = copy.deepcopy(cfg)
        else:
            cfg = compose(self.config_name, overrides)  # type: ignore
        return cfg

    def _get_delta(self, init: DictConfig, other: DictConfig):
        """
        Returns an iterator over all the differences between the init and other config.
        """
        delta = []
        for diff in _compare_config(init, other):
            name = ".".join(diff.path)
            delta.append((name, diff.other_value))
        return delta


def hydra_main(config_name: str, config_path: str, **kwargs):
    """Wrap your main function with this.
    You can pass extra kwargs, e.g. `version_base` introduced in 1.2.
    """
    def _decorator(main: MainFun):
        return HydraMain(main, config_name=config_name, config_path=config_path,
                         **kwargs)
    return _decorator
