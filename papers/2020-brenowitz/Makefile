PKL_DATA=https://github.com/tbeucler/CBRAIN-CAM/raw/master/notebooks/tbeucler_devlog/PKL_DATA
LRFs = lrf/nn_NNAll_20.json lrf/nn_lower_decay_lr_20.json

TOM_BINS = data/2020_03_14_For_Noah.nc
FULL_DATA = ~/workspace/uwnet/data/processed/training/noBlur.nc

all: plots

lrf: $(LRFs)

environment:
	poetry install

data/binned.nc: data/nn_NNAll_20.pkl $(FULL_DATA)
	rm -f $@
	python  bin_data.py $^ $@


lrf/%.json: data/%.pkl
	mkdir -p lrf && python save_jacobian.py $< $@

data/2020_03_14_For_Noah.nc:
	# bash download_tom_data.sh
	python parse_tom_data.py $(PKL_DATA) $@


plots: $(LRFs) data/binned.nc $(TOM_BINS)
	python histogram_plots.py data/binned.nc noah
	python histogram_plots.py \
		--lts-margin=5 \
		--path-margin=16 \
		data/2020_03_14_For_Noah.nc tom
	python bin_plots.py $(TOM_BINS) data/binned.nc
	bash svg_to_pdf.sh
	python wave_structures.py
	python spectra_input_vertical_levels.py

figs/bins.pdf: bin_plots.py $(TOM_BINS)
	python bin_plots.py

figs/fig10.pdf figs/S2.pdf: data/tom_lrfs.json
	python fig10-s2.py $< $@

figs/Figure11.pdf: data/tom_failure_time.json
	python fig11.py $<

figs/S1.pdf: data/s1_data.nc figS1.py 
	python figS1.py $< $@

data/tom_lrfs.json:
	python download_tom_lrfs.py $@

data/tom_failure_time.json:
	python download_fig11_data.py $@

data/s1_data.nc:
	python download_tom_bins_2d.py $(PKL_DATA)/2_27_FigS1_UNSTAB.pkl UNSTAB $@

lrf.json:
	python save_jacobian.py 

standing_instability.pdf: lrf.json
	python standing_instability.py
.PHONY: plots lrf
