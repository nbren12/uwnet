lrf.json:
	python save_jacobian.py

standing_instability.pdf: lrf.json
	python standing_instability.py

data/binned.nc: data/nn_lower_decay_lr_20.pkl
	rm -f $@
	python  bin_data.py $< $@
