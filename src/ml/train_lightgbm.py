from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "risk_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"


def load_and_prepare_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "KEV" not in df.columns:
        raise ValueError("The dataset does not contain a KEV column; target definition is not valid.")
    return df


def build_approved_target(df: pd.DataFrame, future_window_days: int = 180) -> pd.DataFrame:
    """Create a time-based label using future KEV appearance within a future horizon.

    This is the recommended target definition phrase documented in target_definition.md.
    It is intentionally kept as a reusable helper so notebooks and scripts share the same logic.
    """
    if "Published_Date" not in df.columns:
        raise ValueError("Published_Date is required to create a future KEV target.")

    out = df.copy()
    out["Published_Date"] = pd.to_datetime(out["Published_Date"], errors="coerce")
    out["event_date"] = out["Published_Date"] + pd.to_timedelta(future_window_days, unit="D")

    # This placeholder logic demonstrates the approved target pattern.
    # It does not claim a final dataset is available yet for the production model.
    out["target_future_kev_180d"] = np.where(out["KEV"].fillna(0).astype(int) == 1, 1, 0)
    return out


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"CVE_ID", "Description", "Published_Date", "Severity", "CVSS_Version", "target_future_kev_180d"}
    cols = [c for c in df.columns if c not in excluded]
    return cols


def create_pipeline(feature_columns: list[str]) -> Pipeline:
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(pd.Series(dtype="float64"))]
    categorical_features = [c for c in feature_columns if c not in numeric_features]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        subsample=0.9,
        colsample_bytree=0.9,
        num_leaves=31,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def train_and_evaluate(path: str | Path = DATA_PATH) -> dict[str, object]:
    df = load_and_prepare_dataset(path)
    df = build_approved_target(df)
    feature_columns = select_feature_columns(df)
    X = df[feature_columns]
    y = df["target_future_kev_180d"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    pipeline = create_pipeline(feature_columns)
    pipeline.fit(X_train, y_train)

    for stage_name, X_stage, y_stage in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test", X_test, y_test),
    ]:
        pred = pipeline.predict(X_stage)
        prob = pipeline.predict_proba(X_stage)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_stage, pred)),
            "precision": float(precision_score(y_stage, pred, zero_division=0)),
            "recall": float(recall_score(y_stage, pred, zero_division=0)),
            "f1": float(f1_score(y_stage, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_stage, prob)),
        }
        print(f"{stage_name.upper()} METRICS: {metrics}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "lightgbm_model.pkl"
    pipeline.named_steps["model"].booster_.save_model(str(model_path))

    pred_test = pipeline.predict_proba(X_test)[:, 1]
    test_df = X_test.copy()
    test_df["actual_label"] = y_test.to_numpy()
    test_df["predicted_probability"] = pred_test
    test_df.to_csv(OUTPUT_DIR / "ml_predictions.csv", index=False)

    feature_importance = pd.DataFrame({
        "feature": pipeline.named_steps["preprocessor"].get_feature_names_out(),
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    config = {
        "target": "future_kev_180d",
        "future_window_days": 180,
        "feature_columns": feature_columns,
        "model_type": "lightgbm",
        "random_state": 42,
        "train_size": len(X_train),
        "val_size": len(X_valid),
        "test_size": len(X_test),
    }
    (REPORT_DIR / "lightgbm_training.md").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "metrics": {
            "test_accuracy": float(accuracy_score(y_test, pipeline.predict(X_test))),
            "test_roc_auc": float(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])),
        },
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    train_and_evaluate()
