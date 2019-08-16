bucket = "gs://vcm-ml-static/2019-08-19-noah-lrf-paper/"
figures = spectra_input_vertical_levels.pdf

sync: $(figures)
	gsutil cp $(figures) $(bucket)
	gsutil cp index.html $(bucket)
	@echo "access figs at  http://vcm-ml-static.storage.googleapis.com/2019-08-19-noah-lrf-paper/index.html"

%.pdf: %.py
	python $<


