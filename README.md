[![CI Pipeline](https://github.com/PlnThmrs/ECF-POWER/actions/workflows/ci.yml/badge.svg)](https://github.com/PlnThmrs/ECF-POWER/actions/workflows/ci.yml)
# ECF-POWER
Dépôt du projet fil rouge POWER formation M2i-FD1125


## Installation
Le projet nécessite pour fonctionner l'entraînement du modèle de machine à partir des jeux de données collectées non présentes dans ce dépôt.

## Tests
`pytest -v`

## Qualité
- `ruff check .`
- `black --check .`
- `bandit -r .\api\ .\ML_CONSO_regression\src\ .\ML_DPE_regression\src\ -ll`

## Lancement
- Activer Docker (Docker Desktop ou autre)
- Lancer le docker compose : `docker compose up --build`
- Ouvrir le navigateur à l'adresse indiquée par streamlit
