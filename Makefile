.PHONY: venv

download-data:
	kaggle competitions download -c avazu-ctr-prediction -p raw_data

venv:
	pip install -r requirements.txt

training:
	python model.py

submit:
	kaggle competitions submit -c avazu-ctr-prediction -f submission.csv -m "Message"