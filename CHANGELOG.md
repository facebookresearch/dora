# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## Unreleased

Always export RANK and WORLD_SIZE as env variable, so that they can be consumed by Hydra config
resolver.

`dora.log.LogProgress.update` now returns True if logging will happen at the end of the iteration.

Add silent option for grid API, which suppress all printing.

Adding `process_sheep` method in Explorer, that can replace `process_history`.

Automatically simplfies argv list for Hydra experiments when same parameter is repeated multiple times.

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
