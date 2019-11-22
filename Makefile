lrf.json:
	python save_jacobian.py

standing_instability.pdf: lrf.json
	python standing_instability.py

data/binned.nc: data/nn_lower_decay_lr_20.pkl
	rm -f $@
	python  bin_data.py $< $@


plots:
	python histogram_plots.py data/binned.nc noah_bin_plots
	python histogram_plots.py \
		--lts-margin=5 \
		--path-margin=16 \
		data/tom_binned.nc tom_bin_plots

.PHONY: plots
