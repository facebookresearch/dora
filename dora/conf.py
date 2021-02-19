"""
Basic configuration for dora is here.
"""
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
import typing as tp

from .link import Link


@dataclass
class SlurmConfig:
    mem_per_task: float = 10
    gpus: int = 1
    time: int = 1200
    cpus_per_task: int = 10
    partition: str = "learnfair"
    comment: tp.Optional[str] = None
    setup: tp.List[str] = field(default_factory=list)


@dataclass
class DoraDDPConfig:
    rendezvous_file: str = "rendezvous.txt"
    backend: str = "nccl"


@dataclass
class DoraConfig:
    name: str = "default"  # default basename for experiments
    dir: str = "./outputs"  # where everything will be stored
    exclude: tp.List[str] = field(default_factory=list)
    history: str = "history.json"  # where metrics will be stored
    dbs: str = "dbs"  # location inside `dir` with the grid databases
    ddp: DoraDDPConfig = field(default_factory=DoraDDPConfig)

    def is_excluded(self, arg_name):
        for pattern in self.exclude:
            if fnmatch(arg_name, pattern):
                return True
        return False


@dataclass
class DoraRun:
    sig: str
    dora: DoraConfig
    cfg: tp.Any
    argv: tp.List[str]

    link: Link = None

    def __post_init__(self):
        self.link = Link(self)

    @property
    def folder(self) -> Path:
        return Path(self.dora.dir) / self.sig

    @property
    def _argv_cache(self) -> Path:
        return self.folder / ".argv.json"

