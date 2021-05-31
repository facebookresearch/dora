from dora import Explorer
import treetable as tt


class MyExplorer(Explorer):
    def get_grid_metrics(self):
        return [
            tt.leaf("epoch", "d"),
            tt.leaf("train_loss", ".3f"),
            tt.leaf("valid_loss", ".3f"),
            tt.leaf("valid_acc", ".1%"),
        ]


@MyExplorer
def explorer(launcher):
    for bs in [32, 64, 128]:
        for gpus in [1, 2]:
            launcher.slurm(gpus=gpus)(batch_size=bs, dummy=gpus)
