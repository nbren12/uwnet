.PHONY: data docs train reports

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET =gs://uwnet-data
# PROFILE = {{ cookiecutter.aws_profile }}
PROJECT_NAME = uwnet
PYTHON_INTERPRETER = python
GOOGLE_DRIVE_DIR = uwnet-c4290214-d72d-4e2f-943a-d63010a7ecf2
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
data:
	snakemake zonal_time_mean --configfile setup/olympus/config.yaml  -j 20
.PHONY: data

train: #${TRAINING_DATA}
	python -m uwnet.train with assets/training_configurations/default.json  \
         data=data/processed/2018-12-15-longitude-slice-ngaqua.nc \
	     epochs=5 \
		 'training_slices.x=(None,None)'\
         output_dir=rapid_train
train_no_stability_penalty:
	python -m uwnet.train with assets/training_configurations/default.json \
		step.kwargs.alpha=0.0  -m uwnet


train_momentum: ${TRAINING_DATA}
	python -m uwnet.train with data=${TRAINING_DATA} examples/momentum.yaml

run_momentum:
	python src/criticism/run_sam_ic_nn.py \
       -mom-nn models/18/4.pkl \
       -nn models/17/1.pkl \
	     -r \
       data/runs/2018-10-05-q1_q2_and_q3_masked_bndy > 

run_sam:
	python src/criticism/run_sam_ic_nn.py \
		   -nn models/188/5.pkl \
			 -p parameters_sam_neural_network.json \
	      data/runs/2018-11-10-model188-khyp1e6-dt15

sync_data_to_drive:
	rclone sync --stats 5s data/processed $(RCLONE_REMOTE):$(GOOGLE_DRIVE_DIR)/data/processed

upload_reports:
	rsync -avz reports/ olympus:~/public_html/reports/uwnet/

sync_reports:
	rsync -av reports/ ~/public_html/reports/uwnet

setup:  create_environment install_hooks build_image

update_environment:
	conda env update -f environment.yml

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

jupyter:
	docker run -p 8888:8888 -v $(shell pwd):/pwd -w /pwd -v /Users:/Users $(DOCKER_IMAGE) jupyter lab  --port 8888 --ip=0.0.0.0  --allow-root

docs:
	make -C docs html
	ghp-import -n -p docs/_build/html

install_hooks:
	cp -f git-hooks/* .git/hooks/

compile_sam:
	$(MACHINE_SCRIPTS)/compile_sam.sh

build_image:
	docker-compose build sam

enter:
	docker run -w $(shell pwd) -v /home:/home -it nbren12/uwnet:latest bash

test:
	$(MACHINE_SCRIPTS)/run_tests.sh

sync_to_gcs:
	gsutil rsync -r data/processed $(BUCKET)/data/processed
	gsutil rsync -r data/raw $(BUCKET)/data/raw

upload_figs:
	scp notebooks/papers/*.png notebooks/papers/*.pdf \
        olympus:/home/disk/user_www/nbren12/reports/uwnet/plots2019/

.PHONY: test
