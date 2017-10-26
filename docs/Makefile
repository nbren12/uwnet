report.pdf: report.tex
%.pdf: %.tex
	lualatex $<
	biber $*
	lualatex $<
	
plots:
	./generate_plots.sh
