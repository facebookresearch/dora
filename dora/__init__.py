"""

..include:: ../README.md

.. image:: ../dora.jpg

Dora is an experiment management tool which provides the following features:

- Grid search management: automatic scheduling and canceling of the jobs
    to match what is specified in the grid search files. Grid search files
    are pure Python, and can contain arbitrary loops, conditions etc.
- Deduplication: experiments are assigned a signature based on their arguments.
    If you ask twice for the same experiment to be ran, it won't be scheduled twice,
    but merged to the same run. If your code handles checkpointing properly,
    any previous run will be automatically resumed.
- Monitoring: Dora supports basic monitoring from inside the terminal.
    You can customize the metrics to display in the monitoring table,
    and easily track progress, and compare runs in a grid search.

Some Dora concepts:

- A *Grid* is a python file with an explore function, wrapped in a `dora.Explorer`. The explore function takes
    a `dora.Launcher` as argument. Call repeatidly the `dora.Launcher` with a set of
    hyper-parameters to schedule different experiments.
- An *XP* is a specific experiment. Each experiment is defined by the arguments
    passed to the underlying experimental code, and is assigned a signature
    based on those arguments, for easy deduplication.
- A *Sheep* is the association of a Slurm/Submitit job, and an XP.
"""

# flake8: noqa
from .explore import Explorer, Launcher
from .hydra import hydra_main
from .main import argparse_main, get_xp
from .shep import Sheep