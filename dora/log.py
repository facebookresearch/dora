from collections.abc import Iterable, Sized
import logging
import sys
import time
import typing as tp

from treetable.text import colorize


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger: logging.Logger,
                 iterable: Iterable,
                 updates: int = 5,
                 min_interval: int = 1,
                 total: tp.Optional[int] = None,
                 name: str = "LogProgress",
                 level: int = logging.INFO):
        self.iterable = iterable
        if total is None:
            assert isinstance(iterable, Sized)
            total = len(iterable)
        self.total = total
        self.updates = updates
        self.min_interval = min_interval
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            if self.updates > 0:
                log_every = max(self.min_interval, self.total // self.updates)
                # logging is delayed by 1 it, in order to have the metrics from update
                if self._index >= 1 and self._index % log_every == 0:
                    self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.2f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def bold(text: str) -> str:
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


def red(text: str) -> str:
    """Display text in red.
    """
    return colorize(text, "31")


def simple_log(first: str, *args, color=None):
    print(bold(first), *args, file=sys.stderr)


def fatal(*args) -> tp.NoReturn:
    simple_log("FATAL:", *args)
    sys.exit(1)


_dora_handler = None


def setup_logging(verbose=False):
    global _dora_handler  # I know this is dirty
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger('dora')
    logger.setLevel(log_level)
    _dora_handler = logging.StreamHandler(sys.stderr)
    _dora_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
    _dora_handler.setLevel(log_level)
    logger.addHandler(_dora_handler)


def disable_logging():
    assert _dora_handler is not None
    logger = logging.getLogger('dora')
    logger.removeHandler(_dora_handler)
