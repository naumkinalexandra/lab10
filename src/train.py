from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.common import ARTIFACTS_DIR, MODEL_PATH, TARGET_COLUMN


def normalize_target(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    positive = {"1", "1.0", "y", "yes", "approved", "approve", "accepted", "true"}
    negative = {"0", "0.0", "n", "no", "rejected", "reject", "denied", "false"}
    mapped = values.map(lambda v: 1 if v in positive else (0 if v in negative else np.nan))
    if mapped.isna().any():
        raise ValueError(f"Неизвестные значения loan_status: {sorted(values[mapped.isna()].unique())}")
    if mapped.nunique() != 2:
        raise ValueError("loan_status должен содержать два класса.")
    return mapped.astype(int)


def build_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    nums = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    transformers = []
    if nums:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), nums))
    if cats:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cats))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    selector = SelectFromModel(ExtraTreesClassifier(
        n_estimators=80, random_state=42, class_weight="balanced"
    ), threshold="median")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", selector),
        ("classifier", estimator),
    ])


def positive_probability(model, X):
    probs = model.predict_proba(X)
    classes = list(model.named_steps["classifier"].classes_)
    return probs[:, classes.index(1)]


def train(data_path: Path, target: str = TARGET_COLUMN) -> dict:
    data = pd.read_csv(data_path).drop_duplicates().copy()
    if target not in data.columns:
        raise ValueError(f"В CSV отсутствует целевой столбец '{target}'.")
    data = data.dropna(subset=[target])
    y = normalize_target(data.pop(target))
    X = data.drop(columns=[c for c in data.columns if c.lower().startswith("unnamed:")], errors="ignore")
    if X.empty or len(X) < 20:
        raise ValueError("Нужно минимум 20 строк и хотя бы один признак.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=180, min_samples_leaf=2, random_state=42, class_weight="balanced"
        ),
    }
    results, best_name, best_model, best_auc = {}, "", None, -1.0
    for name, estimator in candidates.items():
        fitted = build_pipeline(X_train, estimator).fit(X_train, y_train)
        prob = positive_probability(fitted, X_test)
        pred = fitted.predict(X_test)
        auc = float(roc_auc_score(y_test, prob)) if y_test.nunique() == 2 else float("nan")
        results[name] = {
            "roc_auc": auc,
            "accuracy": float(accuracy_score(y_test, pred)),
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
        }
        if np.isfinite(auc) and auc > best_auc:
            best_name, best_model, best_auc = name, fitted, auc
    if best_model is None:
        raise ValueError("Не удалось рассчитать ROC-AUC на тестовой выборке.")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    metrics = {
        "best_model": best_name, "best_roc_auc": best_auc, "target_column": target,
        "feature_columns": X.columns.tolist(), "train_rows": len(X_train),
        "test_rows": len(X_test), "models": results,
        "note": "Результаты зависят от датасета и не гарантируют качество на новых данных.",
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/loan_data.csv"))
    parser.add_argument("--target", default=TARGET_COLUMN)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.target), ensure_ascii=False, indent=2))
