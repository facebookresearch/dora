import logging

from dora.hydra import main

logger = logging.getLogger(__name__)


@main('config', '../conf')
def main(cfg):
    logger.info(cfg.dora.sig)


if __name__ == "__main__":
    main()
