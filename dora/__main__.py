import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    module = os.environ.get('DORA_PACKAGE')
    parser.add_argument(
        '--package', '-P',
        default=module,
        help='Training module.'
             'To avoid setting this manually, you can also set the DORA_MODULE env variable.')
    parser.add_argument(
        'grid', help='Name of the grid to run.')
