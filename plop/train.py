import argparse
import logging
import sys

from dora import argparse_main, get_run

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser('plop.train')
    parser.add_argument("--lr", default=0.01, type=float)
    return parser


@argparse_main(parser=get_parser())
def main():
    run = get_run()
    print(run.sig, run.folder)


if __name__ == "__main__":
    main()
