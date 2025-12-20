# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lime.lime_tabular import LimeTabularExplainer

# Load models
lr_pipeline = joblib.load("lr_pipeline.joblib")
rf_pipeline = joblib.load("rf_pipeline.joblib")
xgb_pipeline = joblib.load("xgb_tuned_pipeline.joblib")
threshold = joblib.load("best_threshold.joblib")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.sidebar.header("Enter Transaction Features")

# User input function
def user_input_features():
    data = {}
    for col in lr_pipeline.named_steps['preprocess'].transformers_[0][2]:
        data[col] = [st.sidebar.number_input(f"{col}", value=0.0)]
    return pd.DataFrame(data)

input_df = user_input_features()

if st.button("Predict Fraud"):
    lr_prob = lr_pipeline.predict_proba(input_df)[:, 1]
    rf_prob = rf_pipeline.predict_proba(input_df)[:, 1]
    xgb_prob = xgb_pipeline.predict_proba(input_df)[:, 1]

    ensemble_prob = 0.2 * lr_prob + 0.3 * rf_prob + 0.5 * xgb_prob
    pred = (ensemble_prob >= threshold).astype(int)[0]

    st.subheader("Prediction Result")
    st.write("Fraud" if pred == 1 else "Legitimate")
    st.write(f"Fraud Probability: {ensemble_prob[0]:.2%}")

    # Feature Contribution (Top 5)
    X_train_scaled = lr_pipeline.named_steps['preprocess'].transform(pd.read_csv("X_train.csv"))
    lime_explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=input_df.columns,
        class_names=["Legit", "Fraud"],
        mode="classification"
    )

    def predict_fn_wrapper(X_array):
        X_df = pd.DataFrame(X_array, columns=input_df.columns)
        lr_p = lr_pipeline.predict_proba(X_df)[:, 1]
        rf_p = rf_pipeline.predict_proba(X_df)[:, 1]
        xgb_p = xgb_pipeline.predict_proba(X_df)[:, 1]
        return (0.2 * lr_p + 0.3 * rf_p + 0.5 * xgb_p).reshape(-1, 1)

    exp = lime_explainer.explain_instance(input_df.iloc[0].values, predict_fn_wrapper, num_features=5)
    st.subheader("Top 5 Feature Contributions")
    st.write(exp.as_list())
