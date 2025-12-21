# app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

# Configuration
MODEL_DIR = "model"  # adjust if you keep models in a different folder

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.sidebar.header("Enter Transaction Features")

def model_path(fname):
    # prefer model/<fname> if available, otherwise fallback to fname in cwd
    p = os.path.join(MODEL_DIR, fname)
    if os.path.exists(p):
        return p
    if os.path.exists(fname):
        return fname
    return p  # return default path (will error later)

def safe_load(fname):
    p = model_path(fname)
    if not os.path.exists(p):
        st.error(f"Missing file: {p}\nPlease add the required file to the repository.")
        raise FileNotFoundError(p)
    return joblib.load(p)

# Load models (wrapped so we can show helpful error messages)
try:
    lr_pipeline = safe_load("lr_pipeline.joblib")
    rf_pipeline = safe_load("rf_pipeline.joblib")
    xgb_pipeline = safe_load("xgb_tuned_pipeline.joblib")
    threshold_raw = safe_load("best_threshold.joblib")
    threshold = float(np.array(threshold_raw).ravel()[0])
except FileNotFoundError:
    st.stop()  # stop the app; user sees the error message from safe_load

# Robust function to get input feature names
def get_feature_names_from_preprocessor(preproc):
    # Try sklearn >=1.0 API first
    try:
        features = preproc.get_feature_names_out()
        return list(features)
    except Exception:
        pass

    # Fall back to transformers_ attribute (ColumnTransformer)
    try:
        cols = []
        for name, transformer, column_idx_or_names in getattr(preproc, "transformers_", []):
            if isinstance(column_idx_or_names, (list, tuple)):
                cols.extend(list(column_idx_or_names))
            # if it's 'remainder' or 'drop' skip
        if cols:
            return cols
    except Exception:
        pass

    # As a last resort try to read X_train.csv (in model/ or repo root)
    for candidate in [os.path.join(MODEL_DIR, "X_train.csv"), "X_train.csv"]:
        if os.path.exists(candidate):
            try:
                return list(pd.read_csv(candidate).columns)
            except Exception:
                break

    return None

preprocessor = lr_pipeline.named_steps.get("preprocess", None)
feature_names = None
if preprocessor is not None:
    feature_names = get_feature_names_from_preprocessor(preprocessor)

if feature_names is None:
    st.error("Could not determine feature names from the pipeline or X_train.csv. "
             "Ensure your preprocessing step exposes feature names (get_feature_names_out) "
             "or add X_train.csv with column headers to the repository.")
    st.stop()

# User input generation
def user_input_features():
    data = {}
    for col in feature_names:
        # Use number_input for numeric features; adjust ranges/defaults as needed
        data[col] = [st.sidebar.number_input(f"{col}", value=0.0, format="%f")]
    return pd.DataFrame(data)

input_df = user_input_features()

if st.button("Predict Fraud"):
    # Predict probabilities
    try:
        lr_prob = lr_pipeline.predict_proba(input_df)[:, 1]
        rf_prob = rf_pipeline.predict_proba(input_df)[:, 1]
        xgb_prob = xgb_pipeline.predict_proba(input_df)[:, 1]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        st.stop()

    ensemble_prob = 0.2 * lr_prob + 0.3 * rf_prob + 0.5 * xgb_prob
    pred = int((ensemble_prob >= threshold).astype(int)[0])

    st.subheader("Prediction Result")
    st.write("Fraud" if pred == 1 else "Legitimate")
    st.write(f"Fraud Probability: {ensemble_prob[0]:.2%}")

    # Prepare training_data for LIME
    X_train_candidate = None
    for candidate in [os.path.join(MODEL_DIR, "X_train.csv"), "X_train.csv"]:
        if os.path.exists(candidate):
            try:
                X_train_candidate = pd.read_csv(candidate)
                break
            except Exception:
                X_train_candidate = None

    if X_train_candidate is None:
        # try to get a small synthetic sample from existing pipelines if possible
        st.warning("X_train.csv not found; attempting to obtain training data by transforming a small sample.")
        st.info("If LIME explanations fail, add X_train.csv (raw training features) to the repo under model/ or the project root.")
        # If you saved scaler/transformation, you can try to invert or use pipeline internal attributes;
        # here we stop gracefully if no training data is available.
        st.stop()

    try:
        # transform training data through preprocessor if necessary
        X_train_scaled = preprocessor.transform(X_train_candidate)
    except Exception:
        # if transform fails, try to use raw values
        X_train_scaled = X_train_candidate.values

    # LIME Explainer - note: for classification, predict_fn must return shape (n_samples, n_classes)
    lime_explainer = LimeTabularExplainer(
        training_data=np.array(X_train_scaled),
        feature_names=feature_names,
        class_names=["Legit", "Fraud"],
        mode="classification"
    )

    def predict_fn_wrapper(X_array):
        # X_array is raw feature array (n_samples, n_features)
        X_df = pd.DataFrame(X_array, columns=feature_names)
        lr_p = lr_pipeline.predict_proba(X_df)[:, 1]
        rf_p = rf_pipeline.predict_proba(X_df)[:, 1]
        xgb_p = xgb_pipeline.predict_proba(X_df)[:, 1]
        probs = 0.2 * lr_p + 0.3 * rf_p + 0.5 * xgb_p
        # Return two columns: [prob_legit, prob_fraud]
        prob_legit = 1.0 - probs
        prob_fraud = probs
        return np.column_stack([prob_legit, prob_fraud])

    try:
        exp = lime_explainer.explain_instance(input_df.iloc[0].values, predict_fn_wrapper, num_features=5)
        st.subheader("Top 5 Feature Contributions")
        st.write(exp.as_list())
    except Exception as e:
        st.error(f"LIME explanation failed: {e}")
