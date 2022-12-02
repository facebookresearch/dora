# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.12a] - TBD

Fixed bug with PL (Thanks @kingjr).

Added support for the Azure cluster (thanks @JadeCopet).

## [0.1.11] - 2022-09-22

Use job id based seed to avoid systematic failures with port allocation for distributed.

Remove automatic export of WORLD_SIZE inside submitit job target,
use `dora.distrib.set_distrib_env` if you relied on it.

Fixed version_base parameter support that appeared in Hydra.

## [0.1.10] - 2022-06-09

Updated and simplified PyTorch Lightning distributed integration.
Improved overall integration with PL, in particular with PLLogProgress and simplified
Dora logger.

Adding HiPlot support out of the box.

Fixed bug with nested grid searches.

Set `use_rendezvous=False` by default.

More reliable passing of arguments of Hydra (before, setting None would actually fail). I hope this wont break any existing XP sig...

Allow for empty `mem` constraint in Slurm.

Fixing `callbacks` default value in PL.

Extra "keys" in Hydra config files are now allowed (i.e. overrides with `+something=12`).

The package where Dora looks for grids can be customized, in Hydra with `dora.grid_package` in the base config or passing `grid_package='...'` to `argparse_main`.

Better doc for launcher API.

Fix dict support with Hydra. Okay it is time that I release a new version now...

## [0.1.9] - 2022-02-28

Reliable rmtree used to avoid `--clear` being blocked by some locking issues on NFS.

Fix bug with PL.

Early deletion of rendezvous file to avoid errors on job requeue. This might actually lead to
bugs in the future as this is not officially supported but from a discussion with PyTorch engineers,
"it should be okay".

Actually, because rendezvous file are not that reliable, added using Slurm to find
the master addr. Port is decided based on the XP signature (running twice the same XP on the
same machine will crash, but anyway this is probably a bad idea). Set `dora.use_rendezvous: false` to test out. This will soon become the default value.


## [0.1.8] - 2021-12-30

Always export RANK and WORLD_SIZE as env variable, so that they can be consumed by Hydra config
resolver.

`dora.log.LogProgress.update` now returns True if logging will happen at the end of the iteration.

Add silent option for grid API, which suppress all printing.

Adding `process_sheep` method in Explorer, that can replace `process_history` and provide access to the sheep and XP (`sheep.xp`)
in order to allow for processing that can depend on the config of the XP.

Automatically simplifies argv list for Hydra experiments when the same parameter is repeated multiple times.

Better error message when making a typo in the grid name. Always show the traceback when getting an
import error.

Easier sharing of XP hyper params. Added `import`/`export` command to easily share XP hyper-params in text form.
Added shared repository option (`shared` option in Dora config). No metrics or
checkpoints can be shared, this is still a bit dangerous, but this will act as a shared
database for mappings from SIG -> hyper params, so that you can just pass a SIG
to your teammate and launch the same XP. See [the README section on sharing](https://github.com/facebookresearch/dora/blob/main/README.md#sharing-xps) for more details.

## [0.1.7] - 2021-11-08

Adding support for type arrays.

Disabling automatic loading of PyTorch Lightning if installed, as this trigger
a warning with distutils/setuptools.

## [0.1.6] - 2021-10-20

Add py.typed file to source distribution.
Fixed bug in LogProgress for very slow speed.


## [0.1.5] - 2021-09-29

Added possiblity to log always the time per iteration rathen than iterations per seconds.
[PR](https://github.com/facebookresearch/dora/pull/10).

Fixed a bug with `DoraConf`, making sure that the `dir` attribute is always
absolute, even when reset after creation, which could happen when using Hydra
with a relative path for `dora.dir`.


## [0.1.4] - 2021-09-15

Initial release (first versions were private betas).
