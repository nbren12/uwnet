bucket = "gs://vcm-ml-static/2019-08-19-noah-lrf-paper/"
figures = scary_instability.pdf \
	  standing_instability.pdf spectra_input_vertical_levels.pdf

dropbox: $(figures)
	cp $(figures) ~/Dropbox/My\ Articles/InProgress/linear_response_function_paper/figs/

sync: $(figures)
	gsutil cp $(figures) $(bucket)
	gsutil cp index.html $(bucket)
	@echo "access figs at  http://vcm-ml-static.storage.googleapis.com/2019-08-19-noah-lrf-paper/index.html"

%.pdf: %.py
	python $<


