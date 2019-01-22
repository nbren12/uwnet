REMOTE_DIR=olympus:~/public_html/reports/uwnet/plots2019/

plot:
	rm -f 2019-01.html
	jupyter nbconvert --execute ./2019-01.ipynb

upload:

	rsync --delete -av *.png *.pdf $(REMOTE_DIR)

unlock_notebook:
	chmod +w 2019-01.ipynb
	rm -f .~2019-01.ipynb
