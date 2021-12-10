from dora.share import dump, load


def test_dump_load():
    x = [1, 2, 4, {'youpi': 'test', 'b': 56.3}]
    assert load(dump(x)) == x
