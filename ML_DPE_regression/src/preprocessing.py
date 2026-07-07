from __future__ import annotations

import unicodedata  # standardisation des chaînes de caractères pour le mapping catégoriel

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ML_DPE_regression.src.config import RANDOM_STATE, TEST_SIZE
from ML_DPE_regression.src.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TOP_FEATURES,
)
from ML_DPE_regression.src.preprocessing_mappings import MAPPING_TYPE_GENERATEUR

# ============================================================
# 1. MAPPINGS CATEGORIELS
# ============================================================

ETIQUETTE_MAP = {  # %Mapping pour convertir les étiquettes DPE en valeurs numériques
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
}
TYPE_BATIMENT_MAP = {
    "appartement": 0,
    "maison": 1,
}  # Mapping pour convertir les types de bâtiments en valeurs numériques
QUALITE_ISO_MAP = {"insuffisante": 1, "moyenne": 2, "bonne": 3, "très bonne": 4}

MAPPING_PERIODE_TO_ANNEE = (
    {  # Mapping pour convertir les périodes de construction en années approximatives
        "avant 1948": 1923,
        "1948-1974": 1961,
        "1975-1977": 1975,
        "1978-1982": 1980,
        "1983-1988": 1985,
        "1989-2000": 1995,
        "2001-2005": 2003,
        "2006-2012": 2009,
        "2013-2021": 2017,
        "après 2021": 2022,
        "aprÃ¨s 2021": 2022,
    }
)


# ============================================================
# 2. FONCTIONS UTILITAIRES
# ============================================================


# Normalisation des chaînes de caractères pour le mapping catégoriel
def normalize_text(value) -> str:
    if pd.isna(value):
        return "unknown"
    normalized = unicodedata.normalize("NFKD", str(value).lower())
    return "".join(c for c in normalized if not unicodedata.combining(c)).strip()


# Mapping des colonnes catégorielles en valeurs numériques selon les dictionnaires définis ci-dessus
def map_categorical(series: pd.Series, mapping: dict[str, int], default: int = 0):
    return series.map(lambda v: mapping.get(normalize_text(v), default)).astype(float)


# Remplissage des valeurs manquantes pour la colonne "annee_construction" en utilisant la colonne "periode_construction" si disponible
def impute_annee_construction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "periode_construction" in df.columns:
        mask = df["annee_construction"].isna() & df["periode_construction"].notna()
        df.loc[mask, "annee_construction"] = (
            df.loc[mask, "periode_construction"]
            .map(MAPPING_PERIODE_TO_ANNEE)
            .fillna(df["annee_construction"])
        )
    return df


# Remplace les valeurs manquantes dans la colonne "type_generateur_n1_ecs_n1" par une valeur par défaut
def clean_type_generateur(series: pd.Series) -> pd.Series:
    return series.replace(MAPPING_TYPE_GENERATEUR).fillna(
        "Système de production d'ecs non électrique"
    )


# Suppresion des outliers dans les colonnes numériques spécifiées en utilisant la méthode de l'IQR (Interquartile Range)
def remove_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


# ============================================================
# 3. PREPROCESSEUR PRINCIPAL
# ============================================================


# Ensemble des transformations pour préparer les données pour le modèle de régression.
class ProductionPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_fill_values = {}
        self.generator_map = {}
        self.feature_names = TOP_FEATURES.copy()

    def fit(self, X):
        X = self._clean_raw(X)
        X_filled = self._fit_missing_values(X)
        X_encoded = self._fit_encode(X_filled)
        self.scaler.fit(X_encoded)
        return self

    def transform(self, X):
        X = self._clean_raw(X)
        X_filled = self._transform_missing_values(X)
        X_encoded = self._transform_encode(X_filled)
        X_scaled = self.scaler.transform(X_encoded)
        return pd.DataFrame(X_scaled, columns=X_encoded.columns, index=X.index)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # -------------------------
    # Étapes internes
    # -------------------------
    # nettoyage des données brutes et remplissage des années manquantes
    def _clean_raw(self, X):
        X = X.copy()
        X["type_generateur_n1_ecs_n1"] = clean_type_generateur(
            X["type_generateur_n1_ecs_n1"]
        )
        X = impute_annee_construction(X)
        return X

    # remplissage des valeurs manquantes pour les colonnes numériques et catégorielles lors de l'entraînement
    def _fit_missing_values(self, X):
        X = X.copy()
        for col in NUMERIC_FEATURES:
            numeric = pd.to_numeric(X[col], errors="coerce")
            fill_value = float(numeric.mean()) if not numeric.dropna().empty else 0.0
            self.numeric_fill_values[col] = fill_value
            X[col] = numeric.fillna(fill_value)
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].fillna("unknown")
        return X

    # remplissage des valeurs manquantes pour les colonnes numériques et catégorielles lors de la transformation
    def _transform_missing_values(self, X):
        X = X.copy()
        for col in NUMERIC_FEATURES:
            fill_value = self.numeric_fill_values.get(col, 0.0)
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(fill_value)
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].fillna("unknown")
        return X

    # encodage des colonnes catégorielles en valeurs numériques lors de l'entraînement
    def _fit_encode(self, X):
        generator_values = X["type_generateur_n1_ecs_n1"].map(normalize_text)
        self.generator_map = {
            val: i + 1
            for i, val in enumerate(sorted(generator_values.dropna().unique()))
            if val != "unknown"
        }
        return self._transform_encode(X)

    # encodage des colonnes catégorielles en valeurs numériques lors de la transformation
    def _transform_encode(self, X):
        X = X.copy()
        X["etiquette_dpe"] = map_categorical(X["etiquette_dpe"], ETIQUETTE_MAP)
        X["type_batiment"] = map_categorical(
            X["type_batiment"], TYPE_BATIMENT_MAP, default=2
        )
        X["qualite_isolation_enveloppe"] = map_categorical(
            X["qualite_isolation_enveloppe"], QUALITE_ISO_MAP
        )

        X["type_generateur_n1_ecs_n1"] = (
            X["type_generateur_n1_ecs_n1"]
            .map(normalize_text)
            .map(lambda v: self.generator_map.get(v, 0))
            .astype(float)
        )

        return X[TOP_FEATURES].astype(float)


# ============================================================
# 4. PIPELINE COMPLET
# ============================================================


# Construction d'un pipeline complet de prétraitement pour l'entraînement et le test du modèle.
def preprocess_pipeline(X, y, remove_outliers=True):
    missing = [col for col in TOP_FEATURES if col not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if remove_outliers:
        X = remove_outliers_iqr(X, NUMERIC_FEATURES)
        y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = ProductionPreprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor
