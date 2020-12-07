import logging
import sys

from dora.hydra import main

logger = logging.getLogger(__name__)


@main('config', 'conf')
def main(cfg, link):
    link.setup()
    try:
        logger.info(cfg.dora.sig)
    except Exception:
        logger.exception("An error happened")
        sys.exit(1)


if __name__ == "__main__":
    main()
