# Dora The Explorer, a friendly experiment manager

![tests badge](https://github.com/adefossez/dora/workflows/tests/badge.svg)
![linter badge](https://github.com/adefossez/dora/workflows/linter/badge.svg)


<p align="center">
<img src="./dora.jpg" alt="Really cheesy effects applied on a Dora picture."
width="400px"></p>

## Table of Content

- [Installation](#Installation)
- [Introduction](#Introduction)

## Installation

```bash
pip install -U git+https://github.com/facebookincubator/submitit@master#egg=submitit
pip install git+ssh://github.com/fairinternal/dora@main#egg=dora
```

## Introduction

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

Some Dora concepts:

- A *Grid* is a python file with an `explore function`, wrapped in a `dora.Explorer`.
    The explore function takes a `dora.Launcher` as argument. Call repeatidly
    the `dora.Launcher` with a set of
    hyper-parameters to schedule different experiments.
- An *XP* is a specific experiment. Each experiment is defined by the arguments
    passed to the underlying experimental code, and is assigned a signature
    based on those arguments, for easy deduplication.
- A *signature* is the unique XP identifier, derived from its arguments.
    You can use the signature to uniquely identity the XP across runs, and easily
    access logs, checkpoints etc.
- A *Sheep* is the association of a Slurm/Submitit job, and an XP. Given an XP,
    it is always possible to retrieve the last Slurm job that was associated with it.


## Making your code compatible with Dora

Dora supports two backend: `argparse` based code, and `hydra` based code.

For both case, you must have a specific python package (which we will call here `myproj`),
with a `train` module in it, (i.e. `myproj.train` module, stored in the `myproj/train.py` file.)

The `train.py` file must contain a `main` function that is properly decorated, as explained hereafter.

### Argparse support

Here is a template for the `train.py` file:

```python
import argparse
from dora import argparse_main, get_xp

parser = argparse.ArgumentParser("mycode.train")
...


@argparse_main(
    dir="./where_to_store_logs_and_checkpoints",
    parser=parser,
    exclude=["list_of_args_to_ignore_in_signature, e.g.", "num_workers",
             "can_be_pattern_*", "log_*"],
)
def main():
    # No need to reparse args, you can directly access them from the current XP
    # object.
    xp = get_xp()
    xp.sig  # signature for the current run
    xp.folder  # folder for the current run, please put your checkpoint relative
               # to this folder, so that it is automatically resumed!
    xp.link  # link object, can send back metrics to Dora

    for t in range(10):
        xp.link.push_metrics({"loss": 1/(t + 1)})
    ...
```


### Hydra support

The template for `train.py`:
```python
from dora import hydra_main, get_xp


@hydra_main(
    config_path="./conf",  # path where the config is stored, relative to the parent of `mycode`.
    config_name="config"  # a file `config.yaml` should exist there.
)
def main(cfg):
    xp = get_xp()
    xp.sig  # signature for the current run
    # Hydra run folder will automatically be set to xp.folder!

    xp.link  # link object, can send back metrics to Dora

    for t in range(10):
        xp.link.push_metrics({"loss": 1/(t + 1)})
    ...
```

You can customize `dora` behavior from the `config.yaml` file, e.g.

```yaml
my_config: plop
num_workers: 40
logs:
    interval: 10
    level: info

dora:
    exclude: ["num_workers", "logs.*"]
    dir: "./outputs"
```

## Letting Dora find your code

Every Dora command will start with `dora some_command ...`. In order for Dora to find your code, you must pass your training package
(i.e. `mycode`) as `dora -P mycode some_command`.
This flag can be skipped if `mycode` is in the current working directory and is the only folder with a `train.py` file in it, in which
case Dora will find it automatically.
You can also export `DORA_PACKAGE` to avoid having to give the `-P` flag explicitely.


## Distributed training support

Dora supports distributed training, and makes a few assumptions for you. It will schedule one task
per GPU required. You should initialize distributed training through dora, by calling in your `main` function:

```
import dora.distrib
dora.distrib.init()
```

## Running experiments locally

You can debug and run an XP locally with

```bash
dora run [TRAINING_ARGS ...]
```

**Warning**: for the `argparse` backend, you must insert `--` between the dora args and your own training args, i.e.:

```bash
dora run -- [TRAINING_ARGS ...]
```

`dora run` supports two flags:
- `-d`: distributed training using all available gpus. The master worker output will be to the shell, and other workers will be redirected to a log file in the XP folder.
- `-f sig`: this will inject the hyper-parameters from the XP with the given sig on top of the one provided on the command line. Useful to resume locally a remote job that failed.
- `-D, --debug`: will activate the Python debugger when an exception occurs (when combined with
    distributed, only for the master process.)

## Launching experiments remotely

Dora supports scheduling experiments on Slurm. If you need to schedule many of them, then a grid file is properly better.

```dora
dora launch [--dev] [-g NUMBER_OF_GPUS] [TRAINING_ARGS ...]
```

Dora will automatically select the appropriate number of nodes and tasks per nodes based on the number of GPU required, as well as scale required memory.
This command will launch the command, and immediately tail its log and monitor its progress, just like if it were running in locally.
If you want to kill the command if you kill the local process, you can add the `-a`, `--attach` flag.
To avoid tailing the log, just pass `--no_tail`.


If a job already exist for the given XP, Dora will not schedule a second one, but reuse the existing job.

If a previous run has failed or was canceled, Dora will not automatically start a new one, to give you a chance to inspect the logs.
If you want to reschedule a run, use the `-r` flag.

Other flags:
    - `-f SIG`: injects the arguments from the XP matching this signature, on top of the one provided on the command line.
    - `-R, --replace`: replace any running job (i.e. cancels, and schedules a new one).
    - `-D, --replace_done`: also reschedule a job even if a previous one completed successfully.
    - `-p, --partition PARTITION`: partition to use.
    - `-c, --comment COMMENT`: comment for the job (e.g. if priority is used).


## Inspecting an experiment

You can get information on an XP with the `dora info` command:

```bash
dora info [TRAINING_ARGS ...]
dora info -f SIGNATURE
dora info -j SLURM_ID
```

You can either specify the XP by listing all of its training arguments, by passing its signature, or even the latest Slurm id associated with it.
The info command supports a number of flags:
- `-f`: print the folder for the XP
- `-l`: print the entire log for the main task (this only work for remote jobs, not XP ran locally with `dora run`)
- `-t`: tail the log for the main task.

## Grid Files

The main benefit from Dora is the ability to handle arbitarily complex grid searches.
Each *grid* is defined by a grid file, inside a `grids` package (i.e. `mycode.grids.my_grid`).
The grid file



## Advanced configuration

### Setting SLURM default parameters

### Changing the namings of the XPs.


## Developing

Before submitting anychange, please run `make` to run unit tests and code linting.

