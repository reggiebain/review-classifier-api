# Makefile
train:
	python model/train.py

evaluate:
	python model/evaluate.py

serve:
	uvicorn api.main:app --reload

docker-build:
	docker build -t review-api .

docker-run:
	docker run -p 8000:8000 review-api