from dora import Explorer


@Explorer
def explorer(launcher):
    for bs in [32, 64, 128]:
        launcher(batch_size=bs)

    launcher.bind_(gamma=0.6)
    launcher.slurm_(mem_per_gpu=20)
    launcher()
