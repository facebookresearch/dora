
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dora import Explorer
import treetable as tt


class MyExplorer(Explorer):
    def get_grid_metrics(self):
        return [
            tt.leaf("loss", ".3f"),
        ]


@MyExplorer
def explorer(launcher):
    for a in [32, 64, 128]:
        launcher(a=a)

    launcher.bind_(b=4)
    launcher.slurm_(mem_per_gpu=20)
    launcher()

    with launcher.job_array():
        for i in range(100):
            launcher(b=100 + i)
