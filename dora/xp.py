from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
import typing as tp

from .conf import DoraConfig
from .link import Link
from .utils import jsonable


def _get_sig(delta: tp.List[tp.Tuple[str, tp.Any]]) -> str:
    # Return signature from a jsonable content.
    sorted_delta = sorted(delta)
    return sha1(json.dumps(sorted_delta).encode('utf8')).hexdigest()[:8]


@dataclass
class XP:
    """
    Represent a single experiment, i.e. a specific set of parameters
    that is linked to a unique signature.

    One XP can have multiple runs.
    """
    dora: DoraConfig
    cfg: tp.Any
    argv: tp.List[str]
    delta: tp.Optional[tp.List[tp.Tuple[str, tp.Any]]] = None
    sig: tp.Optional[str] = None

    link: Link = None

    def __post_init__(self):
        if self.delta is not None:
            self.delta = jsonable([(k, v) for k, v in self.delta if not self.dora.is_excluded(k)])
        if self.sig is None:
            assert self.delta is not None
            self.sig = _get_sig(self.delta)
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
