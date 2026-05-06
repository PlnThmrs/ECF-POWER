import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def train_model(X_train, y_train):
    """Entraîne trois modèles de régression et retourne le meilleur
    basé sur la MAE en cross-validation."""
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    best_model = None
    best_score = -np.inf

    for _, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    # Entraîne le meilleur modèle sur l'ensemble des données d'entraînement
    best_model.fit(X_train, y_train)
    print(f"Meilleur modèle sélectionné : {type(best_model).__name__}")
    return best_model
