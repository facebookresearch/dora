# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

add py.typed file to source distribution.


## [0.1.5] - 2017-06-20

Added possiblity to log always the time per iteration rathen than iterations per seconds.
[PR](https://github.com/facebookresearch/dora/pull/10).

Fixed a bug with `DoraConf`, making sure that the `dir` attribute is always
absolute, even when reset after creation, which could happen when using Hydra
with a relative path for `dora.dir`.


## [0.1.4] - 2021-09-15

Initial release (first versions were private betas).
