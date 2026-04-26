import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="ResistoScan AI", layout="wide")

st.title("🧬 ResistoScan AI - AMR Surveillance Dashboard")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload ARG dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())

    # -------------------------------
    # BASIC PREPROCESSING
    # -------------------------------
    X = df.drop(columns=["label"], errors='ignore')
    y = df["label"] if "label" in df.columns else np.random.randint(0, 4, len(df))

    # -------------------------------
    # SINGLE MODEL (LOGISTIC)
    # -------------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    pred_lr = lr.predict(X)
    acc_lr = accuracy_score(y, pred_lr)

    # -------------------------------
    # MULTI-MODEL SYSTEM
    # -------------------------------
    rf = RandomForestClassifier()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    svm = SVC(probability=True)
    knn = KNeighborsClassifier()

    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "SVM": svm,
        "KNN": knn
    }

    results = {}

    for name, model in models.items():
        model.fit(X, y)
        pred = model.predict(X)
        acc = accuracy_score(y, pred)
        results[name] = acc

    # -------------------------------
    # REGRESSION MODEL (DUAL SYSTEM)
    # -------------------------------
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression()
    y_reg = np.random.uniform(15000, 25000, len(X))  # dummy ITI values
    reg.fit(X, y_reg)
    iti_pred = reg.predict(X)

    # -------------------------------
    # AMR-ITI CALCULATION
    # -------------------------------
    resistome_density = X.mean(axis=1)
    pathogen_load = np.random.uniform(0.5, 1.0, len(X))
    drug_factor = np.random.uniform(1.0, 1.5, len(X))

    iti_score = resistome_density * (1 + pathogen_load) * drug_factor * 10000

    # -------------------------------
    # DISPLAY RESULTS
    # -------------------------------
    st.subheader("📈 Model Performance")

    st.write(f"Logistic Regression Accuracy: **{acc_lr:.2f}**")

    for name, acc in results.items():
        st.write(f"{name} Accuracy: **{acc:.2f}**")

    # -------------------------------
    # PREDICTIONS
    # -------------------------------
    st.subheader("🔍 Predictions")

    pred_class = lr.predict(X[:1])[0]
    iti_value = iti_score.iloc[0]

    st.write(f"Predicted Environment Class: **{pred_class}**")
    st.write(f"Predicted AMR-ITI Score: **{iti_value:.2f}**")

    # -------------------------------
    # RISK LEVEL
    # -------------------------------
    def risk_level(score):
        if score > 25000:
            return "CRITICAL 🔴"
        elif score > 20000:
            return "VERY HIGH 🟠"
        elif score > 15000:
            return "HIGH 🟡"
        else:
            return "MODERATE 🟢"

    st.subheader("⚠ Risk Assessment")
    st.write(f"Risk Level: **{risk_level(iti_value)}**")

    # -------------------------------
    # SIMULATION
    # -------------------------------
    st.subheader("📊 Predictive Simulation")

    factor = st.slider("ARG Growth Factor", 0.5, 2.0, 1.0)

    simulated_iti = iti_value * factor

    st.write(f"Simulated ITI Score: **{simulated_iti:.2f}**")
    st.write(f"Simulated Risk: **{risk_level(simulated_iti)}**")

else:
    st.info("Please upload a dataset to begin analysis.")
