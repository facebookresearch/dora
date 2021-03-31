from dataclasses import dataclass
from pathlib import Path
import typing as tp

from .conf import DoraConfig
from .link import Link


@dataclass
class XP:
    """
    Represent a single experiment, i.e. a specific set of parameters
    that is linked to a unique signature.

    One XP can have multiple runs.
    """
    sig: str
    dora: DoraConfig
    cfg: tp.Any
    argv: tp.List[str]
    delta: tp.List[tp.Any]

    link: Link = None

    def __post_init__(self):
        self.link = Link(self)

    @property
    def folder(self) -> Path:
        return self.dora.dir / self.dora.xps / self.sig

    @property
    def submitit(self) -> Path:
        return self.folder / self.dora.shep.submitit_folder

    @property
    def rendezvous_file(self) -> Path:
        return self.folder / self.dora.rendezvous_file

    @property
    def history(self) -> Path:
        return self.folder / self.dora.history

    @property
    def _argv_cache(self) -> Path:
        return self.folder / ".argv.json"
