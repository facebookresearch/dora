# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

Always export RANK and WORLD_SIZE as env variable, so that they can be consumed by Hydra config
resolver.

`dora.log.LogProgress.update` now returns True if logging will happen at the end of the iteration.

Add silent option for grid API, which suppress all printing.

Adding `process_sheep` method in Explorer, that can replace `process_history` and provide access to the sheep and XP (`sheep.xp`)
in order to allow for processing that can depend on the config of the XP.

Automatically simplifies argv list for Hydra experiments when the same parameter is repeated multiple times.

Better error message when making a typo in the grid name. Always show the traceback when getting an
import error.

Added `import`/`export` command to easily share XP hyper-params in text form.

Added shared repository option (`shared` option in Dora config). No metrics or
checkpoints can be shared, this is still a bit dangerous, but this will act as a shared
database for mappings from SIG -> hyper params, so that you can just pass a SIG
to your teammate and launch the same XP.

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
