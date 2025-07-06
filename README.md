# Review Classifier API
### A text classification pipeline (train → evaluate → serve)
A FastAPI project to classify text reviews as positive or negative.

- FastAPI endpoints for model prediction
- A test suite with pytest
- A GitHub Actions CI workflow
- Commands via Makefile

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
- `GET /health`

### Package Files Overview
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