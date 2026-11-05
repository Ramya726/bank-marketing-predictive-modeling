from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "y"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(
    page_title="BankData Marketing | Classification ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1450px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.25);}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; background: linear-gradient(115deg,#102a43,#176b87); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; color: white;}
.hero p {font-size: 1rem; margin: 0; color: #e6f4f1;}
.metric-card {border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1rem 1.1rem; min-height: 110px;}
.metric-label {font-size: .85rem; opacity: .8; margin-bottom: .35rem;}
.metric-value {font-size: 1.65rem; font-weight: 700; line-height: 1.2;}
.small-note {font-size: .86rem; opacity: .78;}
</style>
""", unsafe_allow_html=True)


def metric_card(label: str, value: float | None):
    shown = "N/A" if value is None or pd.isna(value) else f"{value:.4f}"
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{shown}</div></div>',
        unsafe_allow_html=True,
    )


def load_available_models():
    return {
        name: MODEL_DIR / filename
        for name, filename in MODEL_FILES.items()
        if (MODEL_DIR / filename).is_file()
    }


def parse_target(series: pd.Series) -> pd.Series:
    # The dataset's target is text; tolerate case and whitespace differences.
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map({"no": 0, "yes": 1, "0": 0, "1": 1})
    if mapped.isna().any():
        examples = cleaned[mapped.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(f"Target '{TARGET}' must contain only yes/no (or 0/1). Found: {examples}")
    return mapped.astype(int)


def auc_score(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("This saved model exposes neither predict_proba nor decision_function for AUC.")


def evaluate_one(name: str, model_path: Path, X: pd.DataFrame, y_true: pd.Series):
    model = joblib.load(model_path)
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        raise ValueError("Saved pipeline does not contain input feature names. Retrain using the supplied train_models.py.")
    missing = [column for column in expected if column not in X.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))
    # Ignore extra columns, but preserve the exact feature order used at training time.
    X_model = X.loc[:, expected]
    y_pred = np.asarray(model.predict(X_model)).astype(int)
    try:
        auc = float(roc_auc_score(y_true, auc_score(model, X_model))) if y_true.nunique() == 2 else np.nan
    except Exception:
        auc = np.nan
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return model, y_pred, metrics, X_model


with st.sidebar:
    st.markdown("## 📈 Bank Marketing")
    st.caption("Bank Marketing · ML Classification")
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Evaluate models", "Compare models", "Dataset & methodology"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Test dataset**")
    uploaded = st.file_uploader(
        "Upload a labelled test CSV", type=["csv"],
        help="CSV must include all model input columns and target column y (yes/no). Upload test data only; models are not retrained here.",
    )
    st.caption("Expected target column: `y` (yes/no)")
    st.divider()
    st.caption("Models are loaded from saved .joblib pipelines. Uploaded data is used only for evaluation.")

st.markdown(
    '<div class="hero"><h1>Bank Marketing Classification</h1>'
    '<p>Evaluate five machine-learning classifiers for term-deposit subscription using a labelled test dataset.</p></div>',
    unsafe_allow_html=True,
)

available_models = load_available_models()
if not available_models:
    st.error("No trained model files were found in the model/ folder.")
    st.markdown("Run `python train_models.py` locally first. It creates the five saved pipelines, `test_data.csv`, and the measured comparison table. Then restart this app.")
    st.stop()

if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded, keep_default_na=False)
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
    data_source = f"Uploaded file: {uploaded.name}"
else:
    bundled_test = ROOT / "test_data.csv"
    if bundled_test.exists():
        try:
            test_df = pd.read_csv(bundled_test, keep_default_na=False)
            data_source = "Bundled test_data.csv (upload another file from the sidebar to replace it)"
        except Exception as exc:
            st.error(f"Could not read bundled test_data.csv: {exc}")
            st.stop()
    else:
        test_df = None
        data_source = "No test data loaded"

if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithms", "5", "Same dataset")
    c2.metric("Evaluation metrics", "6", "Per model")
    c3.metric("Retraining in app", "No", "Saved pipelines only")
    st.subheader("Project at a glance")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("""
        **Objective**

        Predict whether a customer subscribed to a bank term deposit (`y = yes/no`) and compare five classifiers on the same labelled test records.

        **Models included**
        - Logistic Regression
        - Decision Tree Classifier
        - K-Nearest Neighbors
        - Gaussian Naive Bayes
        - Support Vector Machine (LinearSVC)
        """)
    with right:
        st.markdown("### Evaluation workflow")
        st.markdown("1. Train and save models locally.\n2. Upload labelled test CSV.\n3. Select a model or compare all five.\n4. Review metrics and errors.")
        st.info("The `duration` field is excluded because it is only known after the call and would leak information for a pre-call prediction task.")
    st.subheader("Current test data")
    st.write(data_source)
    if test_df is None:
        st.warning("No test CSV is available yet. Run `train_models.py` to generate test_data.csv, then commit it to GitHub for the deployed app.")
    else:
        st.write(f"**{len(test_df):,} rows · {len(test_df.columns)} columns**")
        st.dataframe(test_df.head(8), use_container_width=True)

elif page == "Dataset & methodology":
    st.subheader("Dataset and experimental setup")
    st.markdown("""
    **Dataset:** UCI Bank Marketing (dataset ID 222), a public binary-classification dataset.

    **Target:** `y`; `yes` means the client subscribed to a term deposit, while `no` means they did not.

    **Preprocessing saved inside each pipeline:**
    - Numeric columns: median imputation and standard scaling.
    - Categorical columns: most-frequent imputation and one-hot encoding with unseen categories ignored.
    - Train/test split: stratified 80/20 split with a fixed random seed for reproducibility.
    - `duration` excluded to avoid using information only available after the call.

    **Metrics:** Accuracy, ROC AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).

    **Interpretation note:** The target classes are imbalanced. Accuracy alone can hide poor identification of subscribers, so compare recall, F1, MCC, AUC, and the confusion matrix as well.
    """)
    st.markdown("**Dataset source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)")
    st.warning("Use only the labelled test CSV in this app. It evaluates saved models and does not train or tune them on uploaded data.")

elif page in ("Evaluate models", "Compare models"):
    if test_df is None:
        st.warning("Upload a labelled test CSV using the sidebar, or add the generated test_data.csv to the project root.")
        st.stop()
    if TARGET not in test_df.columns:
        st.error("The CSV must include the target column `y` with values `yes` or `no`.")
        st.stop()
    try:
        y_true = parse_target(test_df[TARGET])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    X = test_df.drop(columns=[TARGET])
    st.caption(f"Data source: {data_source} · {len(test_df):,} labelled rows")

    if page == "Evaluate models":
        model_name = st.selectbox("Choose a classifier", list(available_models.keys()))
        try:
            model, y_pred, metrics, X_used = evaluate_one(model_name, available_models[model_name], X, y_true)
        except Exception as exc:
            st.error(f"Could not evaluate {model_name}: {exc}")
            st.stop()
        st.subheader(f"{model_name} — evaluation summary")
        metric_cols = st.columns(6)
        for col, label in zip(metric_cols, ["Accuracy", "AUC", "Precision", "Recall", "F1-score", "MCC"]):
            with col:
                metric_card(label, metrics[label if label != "F1-score" else "F1"])
        st.divider()
        tab1, tab2, tab3 = st.tabs(["Confusion matrix", "Classification report", "Test data preview"])
        with tab1:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            disp = ConfusionMatrixDisplay(cm, display_labels=["No subscription", "Subscribed"])
            disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
            ax.set_title(model_name)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            tn, fp, fn, tp = cm.ravel()
            a, b, c, d = st.columns(4)
            a.metric("True negatives", f"{tn:,}")
            b.metric("False positives", f"{fp:,}")
            c.metric("False negatives", f"{fn:,}")
            d.metric("True positives", f"{tp:,}")
        with tab2:
            report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["no", "yes"], output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), use_container_width=True)
            st.caption("The positive class is 'yes' (customer subscribed).")
        with tab3:
            preview = X_used.copy()
            preview["Actual y"] = y_true.map({0: "no", 1: "yes"}).values
            preview["Predicted y"] = pd.Series(y_pred, index=preview.index).map({0: "no", 1: "yes"})
            st.dataframe(preview.head(100), use_container_width=True)
            st.download_button("Download predictions CSV", preview.to_csv(index=False), "bank_marketing_predictions.csv", "text/csv")

    else:
        st.subheader("Compare all available classifiers")
        st.write("All models are evaluated on the same uploaded test rows. No model is retrained in this app.")
        metric_rows, failures = [], []
        for name, model_path in available_models.items():
            try:
                _, _, metrics, _ = evaluate_one(name, model_path, X, y_true)
                metric_rows.append(metrics)
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        if failures:
            for failure in failures:
                st.warning(failure)
        if not metric_rows:
            st.error("No model could be evaluated. Check that the CSV has the required input columns.")
            st.stop()
        results = pd.DataFrame(metric_rows).set_index("Model")
        st.dataframe(results.style.format("{:.4f}").highlight_max(axis=0, props="font-weight: bold;"), use_container_width=True)
        st.subheader("Metric comparison")
        selected_metrics = st.multiselect(
            "Metrics to plot", ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
            default=["Accuracy", "AUC", "Recall", "F1", "MCC"],
        )
        if selected_metrics:
            st.bar_chart(results[selected_metrics], use_container_width=True)
        st.download_button("Download comparison CSV", results.reset_index().to_csv(index=False), "bank_marketing_model_comparison.csv", "text/csv")
        st.info("Choose a model under 'Evaluate models' to inspect its confusion matrix and per-class classification report.")
