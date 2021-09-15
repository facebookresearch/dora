# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
import os
import shlex
import subprocess as sp
import typing as tp
from pathlib import Path

from .log import fatal
from .xp import XP


class CommandError(Exception):
    pass


def run_command(command, **kwargs):
    proc = sp.run(command, stdout=sp.PIPE, stderr=sp.STDOUT, **kwargs)
    if proc.returncode:
        command_str = " ".join(shlex.quote(c) for c in command)
        raise CommandError(
            f"Command {command_str} failed ({proc.returncode}): \n" + proc.stdout.decode())
    return proc.stdout.decode().strip()


def check_repo_clean():
    out = run_command(['git', 'status', '--porcelain'])
    clean = out == ""
    if not clean:
        fatal("Repository is not clean! The following files should be commited "
              f"or git ignored: \n {out}")


def get_git_root():
    return Path(run_command(['git', 'rev-parse', '--show-toplevel'])).resolve()


def shallow_clone(target: Path):
    source = get_git_root()
    if target.exists():
        run_command(['git', 'fetch', '--depth=1', 'origin', 'HEAD'], cwd=target)
        run_command(['git', 'checkout', 'FETCH_HEAD'], cwd=target)
    else:
        run_command(['git', 'clone', '--depth=1', source, target])


@contextmanager
def git_save(xp: XP, git_save: bool = True):
    """Context manager that temporarily relocates to a clean clone of the
    current git repository.

    If `git_save` is false, this does nothing.
    """
    if not git_save or '_DORA_ORIGINAL_DIR' in os.environ:
        # if _DORA_ORIGINAL_DIR in env, we already moved to a new folder.
        # if git_save is False, then git saving is disabled.
        yield
    else:
        check_repo_clean()
        target = xp.folder / 'code'
        target.parent.mkdir(exist_ok=True, parents=True)
        shallow_clone(target)

        cwd = Path('.').resolve()
        root = get_git_root()
        relative_path = cwd.relative_to(root)

        os.environ['_DORA_ORIGINAL_DIR'] = str(cwd)
        os.chdir(target / relative_path)
        try:
            yield
        finally:
            os.chdir(cwd)
            del os.environ['_DORA_ORIGINAL_DIR']


def to_absolute_path(path: tp.Union[str, Path]) -> tp.Union[str, Path]:
    """When using `git_save`, this takes a potentially relative path
    with respect to the original execution folder and return an absolute path.
    This is required if you use relative path with respect to this original folder.

    When using both `git_save` and Hydra, two change of directory happens:
    - Dora moves to git clone
    - Hydra moves to XP folder

    Hydra provides a `to_absolute_path()` function. In order to simplify your code,
    if `git_save` was not used, and Hydra is in use, this will fallback to calling
    Hydra version, so that you only need to ever call this function to cover all cases.
    """
    klass = type(path)
    path = Path(path)
    if '_DORA_ORIGINAL_DIR' not in os.environ:
        # We did not use git_save, we check first if Hydra is used,
        # in which case we use it to convert to an absolute Path.
        try:
            import hydra.utils
        except ImportError:
            path = path.resolve()
        else:
            path = hydra.utils.to_absolute_path(str(path))
        return klass(path)
    else:
        # We used git_save, in which case we used the original dir saved by Dora.
        original_cwd = Path(os.environ['_DORA_ORIGINAL_DIR'])
        if path.is_absolute():
            return klass(path)
        else:
            return klass(original_cwd / path)
