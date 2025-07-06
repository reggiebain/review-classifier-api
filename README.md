# Review Classifier API
### A text classification pipeline (train → evaluate → serve)
A FastAPI project to classify text reviews as positive or negative.

## Features
- Logistic regression text classifier
- FastAPI web interface with Swagger docs at `/docs`
- SQLite logging of predictions to `data/predictions.db`
- Auto-retrain endpoint to refresh model on-the-fly

## Getting Started
```bash
make train
make evaluate
make serve
```

## With Docker
```bash
make docker-build
make docker-run
```

## API Endpoints
- `POST /predict` {"text": "some review"}
- `POST /retrain` retrains model from source data and reloads it
- `GET /health`

### Package Files Overview
```
project_root/
├── api/
│   ├── main.py
│   └── model.py
│
├── data/
│   └── raw_reviews.csv  # placeholder
│
├── model/
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── test_api.py
│   └── test_model.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .gitignore
├── Dockerfile
├── Makefile
├── README.md
├── requirements.txt
└── pyproject.toml  # optional for formatting/linting config
```