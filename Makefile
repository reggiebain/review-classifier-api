# Makefile
train:
	python model/train.py

evaluate:
	python model/evaluate.py

serve:
	uvicorn api.main:app --reload

docker-build:
	docker build --progress=plain -t review-api .
	
docker-run:
	docker run -p 8000:8000 review-api

format:
	black .

type-check:
	mypy .

docker-run-dev:
	docker run -p 8000:8000 -v $(pwd)/data:/app/data review-api