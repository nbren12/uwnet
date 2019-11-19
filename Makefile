lrf.json:
	python save_jacobian.py
standing_instability.pdf: lrf.json
	python standing_instability.py
