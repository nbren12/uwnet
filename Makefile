.PHONY: data docs train reports

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET =gs://uwnet-data
# PROFILE = {{ cookiecutter.aws_profile }}
PROJECT_NAME = uwnet
PYTHON_INTERPRETER = python
RCLONE_REMOTE ?= uwgoogledrive
TRAINING_CONFIG=examples/sl_qt.config.yaml
TRAINING_DATA ?= data/processed/2018-10-02-ngaqua-subset.nc
DOCKER_IMAGE ?= nbren12/uwnet:latest
MACHINE ?= docker

MACHINE_SCRIPTS = setup/$(MACHINE)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

WORKDIR = ~/Data/0

upload_reports:
	rsync -avz reports/ olympus:~/public_html/reports/uwnet/

sync_reports:
	rsync -av reports/ ~/public_html/reports/uwnet

setup:  create_environment install_hooks build_image

jupyter:
	docker run -p 8888:8888 -v $(shell pwd):/pwd -w /pwd -v /Users:/Users $(DOCKER_IMAGE) jupyter lab  --port 8888 --ip=0.0.0.0  --allow-root

docs:
	make -C docs html
	ghp-import -n -p docs/_build/html

install_hooks:
	cp -f git-hooks/* .git/hooks/

build_image:
	docker build -t nbren12/uwnet:latest .

enter:
	docker run -w $(shell pwd) -v $(shell pwd):$(shell pwd) -it nbren12/uwnet:latest bash

test:
	$(MACHINE_SCRIPTS)/run_tests.sh

sync_to_gcs:
	gsutil rsync -r data/processed $(BUCKET)/data/processed
	gsutil rsync -r data/raw $(BUCKET)/data/raw

upload_figs:
	scp notebooks/papers/*.png notebooks/papers/*.pdf \
        olympus:/home/disk/user_www/nbren12/reports/uwnet/plots2019/

.PHONY: test
