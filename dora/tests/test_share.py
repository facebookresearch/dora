# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dora.share import dump, load


def test_dump_load():
    x = [1, 2, 4, {'youpi': 'test', 'b': 56.3}]
    assert load(dump(x)) == x
