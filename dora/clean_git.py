import os
import shlex
import subprocess as sp
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
        fatal("Repository is not clean! Please commit all changes. All files should be added to "
              "the repository, or git ignored.")


def get_git_root():
    return Path(run_command(['git', 'rev-parse', '--show-toplevel'])).resolve()


def shallow_clone(target: Path):
    source = get_git_root()
    if target.exists():
        run_command(['git', 'fetch', '--depth=1', 'origin', 'HEAD'], cwd=target)
        run_command(['git', 'checkout', 'FETCH_HEAD'], cwd=target)
    else:
        run_command(['git', 'clone', '--depth=1', source, target])


def get_clone_exec_dir(xp: XP):
    assert xp.dora.clean_git
    root = get_git_root()
    relative_path = Path('.').resolve().relative_to(root)
    return xp.code_folder / relative_path
