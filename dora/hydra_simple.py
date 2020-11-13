from functools import wraps
from hashlib import sha1
import json
import sys

import hydra


def main(**hydra_kwargs):
    def _decorate(_main):
        @wraps(_main)
        def _decorated():
            argv = sys.argv[1:]
            overrides = {}
            for arg in argv:
                if arg.startswith('-'):
                    continue
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    overrides[key] = value
                else:
                    print("bad", arg)
            sig = sha1(json.dumps(sorted(overrides.items())).encode('utf8')).hexdigest()
            sys.argv.append(f"dora.sig={sig}")
            return hydra.main(**hydra_kwargs)(_main)()

        return _decorated
    return _decorate


@main(config_name='config', config_path='../conf')
def test(cfg):
    print(cfg.dora.sig)


if __name__ == "__main__":
    test()
