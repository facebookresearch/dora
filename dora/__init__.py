"""
Dora is an experiment launching tool which provides the following features:

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
    For more advanced monitoring, we support HiPlot out of the box!

Some Dora concepts:

- A *Grid* is a python file with an explore function. The explore function takes
    a `Launcher` as argument. Call repeatidly the `Launcher` with the arguments
    you want to schedule as many experiments.
- An *XP* is a specific experiment. Each experiment is defined by the arguments
    passed to the underlying experimental code, and is assigned a signature
    based on those arguments, for easy deduplication.
- A *Sheep* is the association of a Slurm/Submitit job, and an XP.
"""
# flake8: noqa
from .explore import Explorer
from .hydra import hydra_main
from .main import argparse_main, get_xp
