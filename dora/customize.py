"""
Customizations for Dora.
"""
import json
from pathlib import Path
import typing as tp

from dora.conf import XP, SlurmConfig
import treetable as tt


class Customizations:
    def get_slurm_config(self):
        """Return default Slurm config for the launch and grid actions.
        """
        return SlurmConfig()

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table.
        """
        return tt.group("Metrics", [])

    def get_run_metrics(self, run: XP):
        """Return the metrics for a given run. By default this will look into
        the `history.json` file, that can be populated with the Link class.
        """
        if run.history.exists():
            metrics = json.load(open(run.history))
            return metrics
        else:
            return []

    def short_name_part(self, key: str, value: tp.Any) -> str:
        """Shorten the name of an XP.
        """
        key_parts = key.split(".")
        short_key_parts = []
        for part in key_parts[:-1]:
            short_key_parts.append(part[:3])
        short_key_parts.append(key_parts[-1])
        key = ".".join(short_key_parts)

        if isinstance(value, Path):
            value = value.name
        if value is True:
            return key
        return f"{key}={value}"


custom = Customizations()
