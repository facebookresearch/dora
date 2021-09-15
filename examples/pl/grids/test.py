# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dora import Explorer
import treetable as tt


class MyExplorer(Explorer):
    def get_grid_metrics(self):
        return [tt.group('metrics', [
            tt.leaf("epoch", "d"),
            tt.leaf("train_loss", ".3f"),
            tt.leaf("valid_loss", ".3f"),
            tt.leaf("valid_acc", ".1%"),
        ])]

    def process_history(self, history):
        out = super().process_history(history)
        return {'metrics': out}


@MyExplorer
def explorer(launcher):
    launcher.bind_(max_epochs=10)
    for bs in [32, 64, 128]:
        for gpus in [1, 2]:
            launcher.slurm(gpus=gpus)(batch_size=bs, dummy=gpus)
