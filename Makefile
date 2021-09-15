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

docs:
	pdoc3 --html -o docs -f dora
	cp dora.png docs/dora/

upload: docs
	rsync -ar docs bob:www/share/dora/

live:
	pdoc3 --http : dora


.PHONY: docs tests lint dist clean
