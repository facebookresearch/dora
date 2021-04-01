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
