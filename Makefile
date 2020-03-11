LRFs = lrf/nn_NNAll_20.json lrf/nn_lower_decay_lr_20.json

all: plots

lrf: $(LRFs)

environment:
	poetry install

data/binned.nc: data/nn_lower_decay_lr_20.pkl
	rm -f $@
	python  bin_data.py $< $@


lrf/%.json: data/%.pkl
	mkdir -p lrf && python save_jacobian.py $< $@

data/tom_binned.nc:
	bash download_tom_data.sh
	python parse_tom_data.py


plots: $(LRFs) data/binned.nc data/tom_binned.nc
	python histogram_plots.py data/binned.nc noah
	python histogram_plots.py \
		--lts-margin=5 \
		--path-margin=16 \
		data/tom_binned.nc tom
	python bin_plots.py
	bash svg_to_pdf.sh
	python wave_structures.py
	python spectra_input_vertical_levels.py
	python fig11.py

figs/fig10.pdf: data/tom_lrfs.json
	python fig10.py $<

data/tom_lrfs.json:
	python download_tom_lrfs.py $@

lrf.json:
	python save_jacobian.py 

standing_instability.pdf: lrf.json
	python standing_instability.py
.PHONY: plots lrf
