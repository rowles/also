SHELL := /usr/bin/env bash

define USAGE
Targets
---------------

  clean
  setup_venv
  format
  lint
  utest
  notebook

endef
export USAGE

all:
	@echo "$$USAGE"

clean:
	rm -rf ./venv ./also/*.pyc ./also.egg-info

setup_venv: clean
	python3 -m venv ./venv
	source ./venv/bin/activate \
		&& python3 --version \
		&& pip3 install -e . \
		&& pip3 install -r ./requirements-dev.txt

format:
	source ./venv/bin/activate \
		&& black ./also/ ./setup.py ./tests/

lint:
	source ./venv/bin/activate \
		&& pylint ./also/
utest:
	source ./venv/bin/activate \
		&& py.test ./tests/

notebook:
	source ./venv/bin/activate \
		&& pip3 install matplotlib \
		&& jupyter notebook --NotebookApp.token='' --NotebookApp.password=''

