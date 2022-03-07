# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Internal utilities, likely shouldn't be called from outside.
"""
import os
from pathlib import Path
import sys
import typing as tp

from .log import fatal
from .main import DecoratedMain
from .utils import import_or_fatal


def _find_package(main_module: str):
    cwd = Path(".")
    candidates = []
    for child in cwd.iterdir():
        if child.is_dir() and (child / "__init__.py").exists():
            if (child / f"{main_module}.py").exists():
                candidates.append(child.name)
    if len(candidates) == 0:
        fatal("Could not find a training package. Use -P, or set DORA_PACKAGE to set the "
              "package. Use --main_module or set DORA_MAIN_MODULE to set the module to "
              "be excecuted inside the defined package.")
    elif len(candidates) == 1:
        return candidates[0]
    else:
        fatal(f"Found multiple candidates: {', '.join(candidates)}. "
              "Use -P, or set DORA_PACKAGE to set package being searched. "
              "Use --main_module or set DORA_MAIN_MODULE to set the module being searched "
              "inside the package.")


def get_main(main_module: tp.Optional[str] = None, package: tp.Optional[str] = None):
    if main_module is None:
        main_module = os.environ.get('DORA_MAIN_MODULE') or 'train'
    if package is None:
        package = os.environ.get('DORA_PACKAGE')
        if package is None:
            package = _find_package(main_module)
    module_name = package + "." + main_module
    sys.path.insert(0, str(Path(".").resolve()))
    module = import_or_fatal(module_name)
    try:
        main = module.main
    except AttributeError:
        fatal(f"Could not find function `main` in {module_name}.")

    if not isinstance(main, DecoratedMain):
        fatal(f"{module_name}.main was not decorated with `dora.main`.")
    return main
