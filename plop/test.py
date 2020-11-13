from dora.hydra import main, get_config


@main('config', '../conf')
def main(cfg):
    print(cfg.dora.sig)


if __name__ == "__main__":
    print(get_config('__main__', 'config', '../conf', [], True).hydra.run.dir)
    main()
