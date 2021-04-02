"""

..include:: ../README.md

"""
__pdoc__ = {}
__pdoc__['tests'] = False

# flake8: noqa
from .explore import Explorer, Launcher
from .hydra import hydra_main
from .main import argparse_main, get_xp
from .shep import Sheep

