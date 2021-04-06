"""
This module provides support for Hydra, in particular the `main` wrapper between
the end user `main` function and Hydra.
"""
from collections import namedtuple, OrderedDict
from importlib.util import find_spec
import logging
from pathlib import Path
import sys
import typing as tp

import hydra
from hydra.experimental import compose, initialize_config_dir
from omegaconf.dictconfig import DictConfig

from .conf import DoraConfig, SlurmConfig, update_from_hydra
from .main import DecoratedMain, MainFun, get_xp
from .xp import XP

logger = logging.getLogger(__name__)


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
        assert key in other, "Structure of config should be identical between XPs."
        other_value = other[key]

        if isinstance(ref_value, DictConfig):
            assert isinstance(other_value, DictConfig), \
                "Structure of config should be identical between XPs."
            yield from _compare_config(ref_value, other_value, path)
        elif other_value != ref_value:
            yield _Difference(list(path), key, ref, other, ref_value, other_value)
    assert len(remaining) == 0, "Structure of config should be identical between XPs."
    path.pop(-1)
    return delta


class HydraMain(DecoratedMain):
    def __init__(self, main: MainFun, config_name: str, config_path: str):
        self.config_name = config_name
        self.config_path = config_path

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
        argv = list(argv)
        cfg = self._get_config(argv)
        delta = self._get_delta(self._base_cfg, cfg)
        xp = XP(dora=self.dora, cfg=cfg, argv=argv, delta=delta)
        return xp

    def value_to_argv(self, arg: tp.Any) -> tp.List[str]:
        argv = []
        if isinstance(arg, str):
            argv.append(arg)
        elif isinstance(arg, dict):
            for key, value in arg.items():
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
        run_dir = f"hydra.run.dir={get_xp().folder}"
        sys.argv.append(run_dir)
        try:
            return hydra.main(
                config_name=self.config_name,
                config_path=self.config_path)(self.main)()
        finally:
            sys.argv.remove(run_dir)

    def _get_config(self,
                    overrides: tp.List[str] = [],
                    return_hydra_config: bool = False) -> DictConfig:
        """
        Internal method, returns the config for the given override,
        but without the dora.sig field filled.
        """
        with initialize_config_dir(str(self.full_config_path), job_name=self._job_name):
            return compose(self.config_name, overrides, return_hydra_config=return_hydra_config)

    def _get_delta(self, init: DictConfig, other: DictConfig):
        """
        Returns an iterator over all the differences between the init and other config.
        """
        delta = []
        for diff in _compare_config(init, other):
            name = ".".join(diff.path)
            delta.append((name, diff.other_value))
        return delta


def hydra_main(config_name: str, config_path: str):
    def _decorator(main: MainFun):
        return HydraMain(main, config_name=config_name, config_path=config_path)
    return _decorator
