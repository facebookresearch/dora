import logging

from dora.hydra import main, HydraSupport

logger = logging.getLogger(__name__)


@main('config', '../conf')
def main(cfg):
    logger.info(cfg.dora.sig)


if __name__ == "__main__":
    print(HydraSupport('__main__', 'config', '../conf').get_config([], True).hydra.run.dir)
    main()
