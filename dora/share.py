# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Allow to export and import XP hyper-params using base64 encoded string.
This allows easy sharing through paste, mails etc.
"""
import base64
from functools import partial
import json
import sys
import textwrap
import typing as tp
import zlib


from .main import DecoratedMain
from .log import fatal, simple_log

log = partial(simple_log, "Export:")


def dump(value):
    bits = zlib.compress(json.dumps(value).encode())
    b64 = base64.b64encode(bits)
    return textwrap.fill(b64.decode())


def load(b64):
    b64 = "".join([line.strip() for line in b64.split('\n')])
    bits = base64.b64decode(b64)
    jsoned = zlib.decompress(bits)
    return json.loads(jsoned.decode())


def export_action(args, main: DecoratedMain):
    all_argv = []
    for sig in args.sigs:
        try:
            xp = main.get_xp_from_sig(sig)
        except RuntimeError as error:
            fatal(f"Error loading XP {sig}: {error.args[0]}")
        all_argv.append(xp.argv)
    print()
    print(dump(all_argv))
    print()
    print()


def import_action(args, main: DecoratedMain):
    buffer: tp.List[str] = []
    for line in sys.stdin:
        line = line.strip()
        if not line and buffer:
            break
        if line:
            buffer.append(line)
    all_argv = load("".join(buffer))
    for argv in all_argv:
        xp = main.get_xp(argv)
        main.init_xp(xp)
        name = main.get_name(xp)
        log(f"Imported XP {xp.sig}: {name}")
