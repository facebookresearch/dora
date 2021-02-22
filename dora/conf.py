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
    mem_per_gpu: float = 10
    gpus: int = 1
    time: int = 1200
    cpus_per_gpu: int = 10
    partition: str = "learnfair"
    comment: tp.Optional[str] = None
    setup: tp.List[str] = field(default_factory=list)

    max_num_timeout: int = 20


@dataclass
class ShepConfig:
    job_file: str = "job.pkl"
    by_id: str = "by_id"
    submitit_folder: str = "submitit"


@dataclass
class DoraConfig:
    name: str = "default"  # default basename for experiments
    dir: Path = Path("./outputs")  # where everything will be stored
    exclude: tp.List[str] = field(default_factory=list)
    history: str = "history.json"  # where metrics will be stored
    runs: str = "runs"

    shep: ShepConfig = field(default_factory=ShepConfig)
    rendezvous_file: str = "rendezvous.txt"

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
        return self.dora.dir / self.runs / self.sig

    @property
    def rendezvous_file(self) -> Path:
        return self.folder / self.dora.rendezvous_file

    @property
    def _argv_cache(self) -> Path:
        return self.folder / ".argv.json"

