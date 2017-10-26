report.pdf: report.tex
%.pdf: %.tex
	lualatex $<
	biber $*
	lualatex $<
	
plots:
	jupyter nbconvert --execute plots.ipynb
