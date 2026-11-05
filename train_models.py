"""Train the five required classifiers and save reproducible evaluation artifacts."""
from pathlib import Path
import io
import urllib.request
import zipfile
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
for directory in (MODEL_DIR, RESULTS_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
TARGET = "y"
DROP_FEATURES = ["duration"]  # post-call information; excluded for a pre-call prediction scenario.
RANDOM_STATE = 42
TEST_SIZE = 0.20


def download_dataset() -> Path:
    """Download and unpack the official UCI archive only when needed."""
    local = DATA_DIR / "bank-full.csv"
    if local.exists():
        return local
    print("Downloading UCI Bank Marketing dataset...")
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        nested_name = next((n for n in archive.namelist() if n.lower().endswith("bank.zip")), None)
        if nested_name:
            with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
                csv_name = next(n for n in nested.namelist() if n.lower().endswith("bank-full.csv"))
                local.write_bytes(nested.read(csv_name))
        else:
            csv_name = next(n for n in archive.namelist() if n.lower().endswith("bank-full.csv"))
            local.write_bytes(archive.read(csv_name))
    return local


def score_for_auc(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def calculate_metrics(name, y_true, y_pred, y_score):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    source_path = download_dataset()
    df = pd.read_csv(source_path, sep=";", keep_default_na=False)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the UCI CSV.")

    y = df[TARGET].astype(str).str.strip().str.lower().map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected values found in target column y.")
    drop_cols = [TARGET] + [column for column in DROP_FEATURES if column in df.columns]
    X = df.drop(columns=drop_cols)
    if X.shape[1] < 12 or len(df) < 500:
        raise ValueError("Dataset does not satisfy the assignment minimum of 12 features and 500 rows.")

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [column for column in X.columns if column not in numeric_cols]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    classifiers = {
        "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        "decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE)),
        "knn": ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "naive_bayes": ("Gaussian Naive Bayes", GaussianNB()),
        "svm": ("Support Vector Machine", LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=10000)),
    }

    rows = []
    for key, (display_name, classifier) in classifiers.items():
        print(f"\nTraining {display_name}...")
        # A fresh preprocessor is created for each model to avoid accidental fitted-state sharing.
        model_preprocessor = ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical_cols),
        ], remainder="drop")
        pipeline = Pipeline([("preprocessor", model_preprocessor), ("classifier", classifier)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = score_for_auc(pipeline, X_test)
        rows.append(calculate_metrics(display_name, y_test, y_pred, y_score))
        model_path = MODEL_DIR / f"{key}.joblib"
        joblib.dump(pipeline, model_path, compress=3)
        print(f"Saved {model_path.name}")

    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    # Labelled test CSV is required by the assignment and used by Streamlit for evaluation.
    test_export = X_test.copy()
    test_export[TARGET] = y_test.map({0: "no", 1: "yes"})
    test_export.to_csv(ROOT / "test_data.csv", index=False)
    X_test.head(100).to_csv(ROOT / "test_data_without_labels.csv", index=False)

    print("\n=== Evaluation metrics on the shared stratified test split ===")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved labelled test data: {ROOT / 'test_data.csv'}")
    print(f"Saved comparison results: {RESULTS_DIR / 'model_comparison.csv'}")
    print("Training complete. The Streamlit app loads these saved pipelines; it does not retrain them.")


if __name__ == "__main__":
    main()
