from contextlib import contextmanager
import typing as tp
from unittest import mock

import submitit

from ..shep import Shepherd


class FakeWatcher:
    def __init__(self):
        self.jobs = {}

    def get_state(self, job_id: str, mode: str = "standard") -> str:
        return self.jobs[job_id]

    def is_done(self, job_id: str, mode: str = "standard") -> bool:
        return self.get_state(job_id) in ["COMPLETED", "FAILED", "CANCELLED"]


class FakeJob:
    watcher = FakeWatcher()

    def __init__(self):
        self.job_id = str(len(self.watcher.jobs))
        self._state = 'UNKNOWN'

    @property
    def _state(self):
        return self.watcher.get_state(self.job_id)

    @_state.setter
    def _state(self, state: str):
        self.watcher.jobs[self.job_id] = state


class FakeExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def update_parameters(self, **kwargs):
        pass

    def submit(self, *args, **kwargs) -> FakeJob:
        return FakeJob()


def _fake_cancel(self, jobs: tp.List[FakeJob]):
    for job in jobs:
        job._state = "CANCELLED"


@contextmanager
def mock_shep():
    with mock.patch.object(submitit, "SlurmExecutor", FakeExecutor):
        with mock.patch.object(Shepherd, "_cancel", _fake_cancel):
            try:
                yield
            finally:
                FakeJob.watcher.jobs = {}
