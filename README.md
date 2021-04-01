# Dora The Explorer, a friendly experiment manager

<p align="center">
<img src="./dora.jpg" alt="Really cheesy effects applied on a Dora picture."
width="600px"></p>

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

You can debug and run everything locally by calling

```bash
dora run [ARGS ...]
```

**Warning**: for the `argparse` backend, you must insert `--` between the dora args and your own training args, i.e.:

```bash
dora [-P mycode] run -- [ARGS ...]
```
### run flags

`dora run` supports two flags:
- `-d`: distributed training using all available gpus. The master worker output will be to the shell, and other workers will be redirected to a log file in the XP folder.
- `-f sig`: this will inject the hyper-parameters from the XP with the given sig on top of the one provided on the command line. Useful to resume locally a remote job that failed.

## Launching experiments remotely

Dora supports scheduling experiments on Slurm. If you need to schedule many of them, then a grid file is properly better.

```dora
dora

