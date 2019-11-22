LRFs = lrf/nn_NNAll_20.json lrf/nn_lower_decay_lr_20.json


lrf: $(LRFs)

data/binned.nc: data/nn_lower_decay_lr_20.pkl
	rm -f $@
	python  bin_data.py $< $@


lrf/%.json: data/%.pkl
	mkdir -p lrf && python save_jacobian.py $< $@

plots: $(LRFs)
	python histogram_plots.py data/binned.nc noah
	python histogram_plots.py \
		--lts-margin=5 \
		--path-margin=16 \
		data/tom_binned.nc tom
	bash svg_to_pdf.sh
	python wave_structures.py
	python spectra_input_vertical_levels.py


lrf.json:
	python save_jacobian.py 

standing_instability.pdf: lrf.json
	python standing_instability.py
.PHONY: plots lrf
