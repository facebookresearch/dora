import typing as tp

from .main import DecoratedMain
from .shep import Sheep


def get_names(herd: tp.List[Sheep], main: DecoratedMain) -> tp.List[str]:
    reference = main.get_name_parts(herd[0])
    name_parts = []
    for xp in herd:
        np = main.get_name_parts(xp)
        for key, val in np.items():
            if key in reference and reference[key] != val:
                reference.pop(key)

        name_parts.append(np)

    names = []
    for np in name_parts:
        for key in reference.keys():
            np.pop(key)
        name = dora_config.get_short_name(np)

    base_name = dora_config.get_short_name(reference)

