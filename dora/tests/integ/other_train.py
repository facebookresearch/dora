# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

from dora.tests.test_main import get_main

main = get_main(os.environ['_DORA_TEST_TMPDIR'])
