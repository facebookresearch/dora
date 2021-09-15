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
            tt.leaf("train", ".3f"),
            tt.leaf("test", ".3f"),
            tt.leaf("correct", ".1f"),
        ]


@MyExplorer
def explorer(launcher):
    for bs in [32, 64, 128]:
        launcher(batch_size=bs)

    launcher.bind_(gamma=0.6)
    launcher.slurm_(mem_per_gpu=20)
    launcher()
