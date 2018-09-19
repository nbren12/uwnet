.PHONY: data docs train

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = {{ cookiecutter.s3_bucket }}
# PROFILE = {{ cookiecutter.aws_profile }}
PROJECT_NAME = uwnet
PYTHON_INTERPRETER = python
GOOGLE_DRIVE_DIR = uwnet-c4290214-d72d-4e2f-943a-d63010a7ecf2
RCLONE_REMOTE ?= uwgoogledrive
TRAINING_CONFIG=examples/sl_qt.config.yaml
TRAINING_DATA ?=data/processed/tropics.nc

#################################################################################
# COMMANDS                                                                      #
#################################################################################

WORKDIR = ~/Data/0

./nextflow:
	wget -qO- https://get.nextflow.io | bash

all: ./nextflow
	./nextflow run main.nf -w $(WORKDIR)  -resume \
		 --numTimePoints=640 --forcingMethod=FD

no_resume:
	./nextflow run main.nf -w $(WORKDIR) --forcingMethod=FD

debug_sam_output: 
	rm -f ~/workspace/models/SAMUWgh/SAM_*
	make all
	@grep 'W range' data/samNN/checks/sam_nn.txt
	@grep 'FQT range' data/samNN/checks/sam_nn.txt
	@grep -v 'FQT range' data/samNN/checks/sam_nn.txt | grep FQT
	@grep FSL data/samNN/checks/sam_nn.txt
	@grep Prec data/samNN/checks/sam_nn.txt

check_sam_output:
	make all
	@grep 'W range' data/samNN/checks/sam_nn.txt
	@grep 'FQT range' data/samNN/checks/sam_nn.txt
	@grep -v 'FQT range' data/samNN/checks/sam_nn.txt | grep FQT
	@grep FSL data/samNN/checks/sam_nn.txt
	@grep Prec data/samNN/checks/sam_nn.txt

print_sam_checks:
	@grep 'W range' data/samNN/checks/sam_nn.txt
	@grep 'FQT range' data/samNN/checks/sam_nn.txt
	@grep -v 'FQT range' data/samNN/checks/sam_nn.txt | grep FQT
	@grep FSL data/samNN/checks/sam_nn.txt
	@grep Prec data/samNN/checks/sam_nn.txt

## Call nextflow to produce the training data.
${TRAINING_DATA}:
	snakemake data/processed/training.nc

## train
train: ${TRAINING_DATA}
	python -m uwnet.train with data=${TRAINING_DATA} examples/sl_qt.config.yaml -m uwnet


sync_data_to_drive:
	rclone sync --stats 5s data/processed $(RCLONE_REMOTE):$(GOOGLE_DRIVE_DIR)/data/processed

upload_reports:
	rsync -avz reports/ olympus:~/public_html/reports/uwnet/

docker:
	docker run -it \
		-v /Users:/Users  \
		-v $(shell pwd)/uwnet:/opt/uwnet \
		-v $(shell pwd)/ext/sam:/opt/sam \
		-w $(shell pwd) nbren12/uwnet bash

build_image:
	docker build -t nbren12/uwnet .

setup:  create_environment install_hooks build_image

create_environment:
	@echo ">>> creating environment from file"
	conda env create -f environment.yml || \
	conda env update -f environment.yml
	@echo ">>> Setting up uwnet in develop mode"
	bash -c "source activate uwnet && python setup.py develop && \
           pip install -e ext/sam/SCRIPTS/python && \
					 jupyter labextension install @pyviz/jupyterlab_pyviz"
	@echo ">>> New Environment created...activate by typing"
	@echo "    source activate uwnet"

docs:
	make -C docs html

install_hooks:
	cp -f git-hooks/* .git/hooks/
