# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path

from setuptools import setup, find_packages

NAME = 'dora_search'
DESCRIPTION = 'Easy grid searches for ML.'

URL = 'https://github.com/facebookresearch/dora'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez'
REQUIRES_PYTHON = '>=3.7.0'

for line in open('dora/__init__.py'):
    line = line.strip()
    if '__version__' in line:
        context = {}
        exec(line, context)
        VERSION = context['__version__']

HERE = Path(__file__).parent

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    package_data={"dora": ["py.typed"]},
    install_requires=['omegaconf', 'retrying', 'submitit', 'treetable', 'torch'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['dora=dora.__main__:main'],
    },
    extras_require={
        'dev': ['coverage', 'flake8', 'hiplot', 'hydra-core', 'hydra_colorlog',
                'mypy', 'pdoc3', 'pytest', 'pytorch_lightning'],
    },
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
