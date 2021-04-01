all: tests lint

lint:
	flake8 dora && mypy -p dora

tests:
	coverage run -m pytest || exit 1
	coverage report --include 'dora/*'
	coverage html --include 'dora/*'

dist:
	python3 setup.py sdist

clean:
	rm -r docs dist build *.egg-info



.PHONY: tests lint dist clean
