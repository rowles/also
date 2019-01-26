SHELL := /usr/bin/env bash

all:
	echo "blah"

clean:
	rm -rf ./venv ./also/*.pyc

setup_venv: clean
	python3 -m venv ./venv
	source ./venv/bin/activate \
		&& python3 --version \
		&& pip3 install -e . \
		&& pip3 install -r ./requirements-dev.txt

format:
	source ./venv/bin/activate \
		&& black ./also/ ./setup.py

lint:
	source ./venv/bin/activate \
		&& pylint ./also/
utest:
	source ./venv/bin/activate \
		&& py.test --cov=also ./tests/


