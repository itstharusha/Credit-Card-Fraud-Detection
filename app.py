import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer
from sklearn.inspection import permutation_importance

# =====================================================
# App Config
# =====================================================
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Platform")
st.caption("Ensemble ML • Explainable AI • Production-Ready")

# =====================================================
# Load Models
# =====================================================
@st.cache_resource
def load_assets():
    lr = joblib.load("model/lr_pipeline.joblib")
    rf = joblib.load("model/rf_pipeline.joblib")
    xgb = joblib.load("model/xgb_tuned_pipeline.joblib")

    with open("model/ensemble_threshold.txt") as f:
        threshold = float(f.read())

    return lr, rf, xgb, threshold


lr_model, rf_model, xgb_model, THRESHOLD = load_assets()

# =====================================================
# Sidebar Inputs
# =====================================================
st.sidebar.header("🧾 Transaction Details")

def get_user_input():
    data = {}
    for i in range(1, 29):
        data[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0)

    data["Amount"] = st.sidebar.number_input("Transaction Amount", value=120.0)
    data["Time"] = st.sidebar.number_input("Time (seconds)", value=100000.0)

    return pd.DataFrame([data])


X_input = get_user_input()

# =====================================================
# Prediction Logic
# =====================================================
lr_prob = lr_model.predict_proba(X_input)[:, 1][0]
rf_prob = rf_model.predict_proba(X_input)[:, 1][0]
xgb_prob = xgb_model.predict_proba(X_input)[:, 1][0]

ensemble_prob = (
    0.2 * lr_prob +
    0.3 * rf_prob +
    0.5 * xgb_prob
)

prediction = int(ensemble_prob >= THRESHOLD)

# Risk Banding
if ensemble_prob < 0.3:
    risk = "LOW"
elif ensemble_prob < 0.7:
    risk = "MEDIUM"
else:
    risk = "HIGH"

# =====================================================
# Tabs Layout
# =====================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Prediction Summary",
    "🧠 Local Explanation (LIME)",
    "🌍 Global Feature Importance"
])

# =====================================================
# TAB 1 — Summary
# =====================================================
with tab1:
    st.subheader("Model Decision Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud Probability", f"{ensemble_prob:.4f}")
    col2.metric("Risk Level", risk)
    col3.metric("Decision Threshold", f"{THRESHOLD:.2f}")

    st.divider()

    if prediction == 1:
        st.error("🚨 High-risk transaction flagged as FRAUD")
    else:
        st.success("✅ Transaction classified as LEGITIMATE")

    st.subheader("Ensemble Breakdown")
    breakdown = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Fraud Probability": [lr_prob, rf_prob, xgb_prob],
        "Weight": [0.2, 0.3, 0.5]
    })

    st.dataframe(breakdown, use_container_width=True)

# =====================================================
# TAB 2 — LIME
# =====================================================
with tab2:
    st.subheader("Local Explanation (Why this prediction?)")

    explainer = LimeTabularExplainer(
        training_data=np.array(X_input),
        feature_names=X_input.columns,
        class_names=["Legit", "Fraud"],
        mode="classification"
    )

    explanation = explainer.explain_instance(
        X_input.iloc[0].values,
        xgb_model.predict_proba,
        num_features=10
    )

    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)

    st.subheader("Top Feature Contributions")
    lime_df = pd.DataFrame(
        explanation.as_list(),
        columns=["Feature", "Impact"]
    )

    st.dataframe(lime_df, use_container_width=True)

# =====================================================
# TAB 3 — Global Importance
# =====================================================
with tab3:
    st.subheader("Global Feature Importance (Permutation)")

    st.info("Computed on synthetic batch for demonstration purposes")

    if st.button("Run Permutation Importance"):
        with st.spinner("Computing..."):
            sample = pd.concat([X_input] * 100, ignore_index=True)

            r = permutation_importance(
                xgb_model,
                sample,
                np.zeros(len(sample)),
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )

            importance_df = pd.DataFrame({
                "Feature": X_input.columns,
                "Importance": r.importances_mean
            }).sort_values("Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(
                importance_df.head(10)["Feature"],
                importance_df.head(10)["Importance"]
            )
            ax.invert_yaxis()
            ax.set_title("Top Global Features")

            st.pyplot(fig)
            st.dataframe(importance_df.head(15), use_container_width=True)

# =====================================================
# Footer
# =====================================================
st.divider()
st.caption(
    "Built with scikit-learn, XGBoost & Streamlit • Designed for real-world deployment"
)

