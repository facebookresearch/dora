#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
# Inspired from https://github.com/kennethreitz/setup.py

from pathlib import Path

from setuptools import setup, find_packages

NAME = 'dora'
DESCRIPTION = 'Easy grid searches for ML'

URL = 'https://github.com/fairinternal/dora'
EMAIL = 'defossez@fb.com'
AUTHOR = 'Alexandre Défossez'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = "0.1.4"

HERE = Path(__file__).parent

REQUIRED = [i.strip() for i in open("requirements.txt") if '/' not in i]

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
        'dev': ['coverage', 'flake8', 'hydra-core', 'hydra_colorlog',
                'mypy', 'pdoc3', 'pytest', 'pytorch_lightning'],
    },
    license='Creative Commons Attribution-NonCommercial 4.0 International',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
