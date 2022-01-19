# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez 2020

from contextlib import contextmanager
import importlib
import logging
import os
from pathlib import Path
import pickle
from shutil import rmtree
import tempfile
import typing as tp

from omegaconf.basecontainer import BaseContainer
from omegaconf import OmegaConf

from .log import fatal

logger = logging.getLogger(__name__)


def jsonable(value):
    import torch

    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    elif isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    elif isinstance(value, Path):
        return str(value)
    elif value is None or isinstance(value, (int, float, str, bool)):
        return value
    elif isinstance(value, BaseContainer):
        return OmegaConf.to_container(value)
    else:
        raise ValueError(f"{repr(value)} is not jsonable.")


@contextmanager
def write_and_rename(path: Path, mode: str = "wb", suffix: str = ".tmp"):
    """
    Write to a temporary file with the given suffix, then rename it
    to the right filename. As renaming a file is usually much faster
    than writing it, this removes (or highly limits as far as I understand NFS)
    the likelihood of leaving a half-written checkpoint behind, if killed
    at the wrong time.
    """
    tmp_path = str(path) + suffix
    with open(tmp_path, mode) as f:
        yield f
    os.rename(tmp_path, path)


def try_load(path: Path, load=pickle.load, mode: str = "rb"):
    """
    Try to load from a path using torch.load, and handles various failure cases.
    Return None upon failure.
    """
    try:
        return load(open(path, mode))
    except (OSError, pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        # Trying to list everything that can go wrong.
        logger.warning(
            "An error happened when trying to load from %s, this file will be ignored: %r",
            path, exc)
        return None


def import_or_fatal(module_name: str) -> tp.Any:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        logger.info("Could not import module %s", module_name, exc_info=True)
        fatal(f"Failed to import module {module_name}.")


def reliable_rmtree(path: Path):
    """Reliably delete the given folder, trying to remove while ignoring errors,
    and if any files remain, renaming to some trash folder."""
    error = False

    def _on_error(func, error_path, exc_info):
        nonlocal error
        error = True
        logger.warning(f"Error deleting file {error_path}")

    rmtree(path, onerror=_on_error)
    if error:
        assert path.exists()
        target_name = tempfile.mkdtemp(dir=path.parent, prefix=path.name + "_", suffix="_trash")
        logger.warning(f"Deletion of {path} failed, moving to {target_name}")
        path.rename(target_name)
    else:
        assert not path.exists()
