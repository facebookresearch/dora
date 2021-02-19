# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez 2020

from contextlib import contextmanager
import functools
import inspect
import logging
from pathlib import Path
import pickle
import os
import weakref

import torch

logger = logging.getLogger(__name__)

_init_cache = weakref.WeakKeyDictionary()


def capture_init(klass):
    """capture_init.

    Decorate a class with this , and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """

    old_init = klass.__init__

    @functools.wraps(old_init)
    def __init__(self, *args, **kwargs):
        _init_cache[self] = (args, kwargs)
        old_init(self, *args, **kwargs)

    klass.__init__ = __init__
    return klass


def serialize_model(model):
    try:
        args, kwargs = _init_cache[model]
    except KeyError:
        raise ValueError(
            f"Couldn't find the saved args and kwargs for {model}, did you use `@capture_init`?")
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


def deserialize_model(package, model=None, strict_args=False, strict_params=False):
    """
    deserialize_model.
    """
    if model is None:
        klass = package['class']
        if strict_args:
            model = klass(*package['args'], **package['kwargs'])
        else:
            sig = inspect.signature(klass)
            kw = package['kwargs']
            for key in list(kw):
                if key not in sig.parameters:
                    logger.warning("Dropping inexistant parameter %s", key)
                    del kw[key]
            model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'], strict=strict_params)
    return model


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def get_metric(metrics, name):
    parts = name.split(".")
    current = metrics
    for part in parts:
        try:
            current = current[part]
        except KeyError:
            raise KeyError(f"Failed to get key {name}")
    return current


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(get_metric(history, name))
    return out


def jsonable(value):
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
    else:
        raise ValueError(f"{value:r} is not jsonable.")


@contextmanager
def write_and_rename(path: Path, suffix: str = ".tmp", mode: str = "wb"):
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


def try_load(path: Path, load=torch.load, mode="rb"):
    """
    Try to load from a path using torch.load, and handles various failure cases.
    Return None upon failure.
    """
    try:
        return load(open(path, mode))
    except (IOError, OSError, pickle.UnpicklingError, RuntimeError, EOFError) as exc:
        # Trying to list everything that can go wrong.
        logger.warning(
            "An error happened when trying to load from %s, this file will be ignored: %r",
            path, exc)
        return None
