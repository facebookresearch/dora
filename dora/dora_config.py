"""
Default dora config.
"""
import json
from pathlib import Path

from dora.conf import XP, SlurmConfig
import treetable as tt


def get_slurm_config():
    return SlurmConfig()


def get_grid_metrics():
    return tt.group("Metrics", [])


def get_run_metrics(run: XP):
    if run.history.exists():
        metrics = json.load(open(run.history))
        return metrics
    else:
        return []


def shorten_name_part(key, value):
    key_parts = key.split(".")
    short_key_parts = []
    for part in key_parts[:-1]:
        short_key_parts.append(part[:3])
    short_key_parts.append(key_parts[-1])
    key = ".".join(short_key_parts)

    if isinstance(value, Path):
        value = value.name
    return (key, value)


def chain_modules(*modules):
    pass
