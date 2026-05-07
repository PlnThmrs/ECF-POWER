[![CI Pipeline](https://github.com/Speedyv01/ml-filrouge-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Speedyv01/ml-filrouge-project/actions/workflows/ci.yml)

# Projet ML industrialisé

## Installation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Tests
pytest -v

## Qualité
ruff check .
black --check .
bandit -r src -ll