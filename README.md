# Review Classifier API

A lightweight, production-ready sentiment analysis API built with FastAPI and scikit-learn.

## 🚀 Features
- Predict sentiment from single or multiple review texts
- Log predictions in a local SQLite database
- RESTful API with automatic Swagger UI
- Dockerized for portability
- CI pipeline with type checking, linting, and test coverage

## 📦 Project Structure
```bash
├── api/                # FastAPI endpoints and model logic
├── model/              # Model training and evaluation code
├── tests/              # Unit tests
├── data/               # SQLite database (auto-generated)
├── docs/               # Markdown developer and API documentation
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker container config
├── Makefile            # Common dev tasks
└── .github/workflows/  # GitHub Actions CI config
```
## Getting Started
### Local Development
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make train
make serve
```
Visit the API at: http://localhost:8000/docs

### With Docker
```
make docker-build
make docker-run
```

## CI/CD
Runs on push to GitHub:

- Code formatting with Black

- Type checking with MyPy

- Model training

- Unit tests with pytest + coverage

