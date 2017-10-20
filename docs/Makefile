report.pdf: report.tex
%.pdf: %.tex
	lualatex $<
	biber $*
	lualatex $<
	
