import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix

st.set_page_config(layout="wide")

st.title("🧬 AMR Surveillance Dashboard (AI + Explainable ML)")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("../data/iti_scores.csv")

# ==============================
# FILTER
# ==============================
env_list = df["Environment"].unique()
selected_env = st.sidebar.multiselect("Environment", env_list, default=env_list)
filtered_df = df[df["Environment"].isin(selected_env)]

# ==============================
# METRICS
# ==============================
col1, col2, col3 = st.columns(3)
col1.metric("Samples", len(filtered_df))
col2.metric("Avg ITI", round(filtered_df["ITI_Score"].mean(), 2))
col3.metric("Max ARGs", int(filtered_df["Total_ARGs"].max()))

# ==============================
# PLOTS
# ==============================
st.plotly_chart(px.bar(filtered_df, x="Sample_ID", y="ITI_Score", color="Environment"))
st.plotly_chart(px.scatter(filtered_df, x="Total_ARGs", y="ITI_Score", color="Environment"))

# ==============================
# HEATMAP
# ==============================
fig, ax = plt.subplots()
sns.heatmap(filtered_df[["Total_ARGs","ITI_Score","High_Priority_ARGs"]].corr(), annot=True, ax=ax)
st.pyplot(fig)

# ==============================
# ML MODEL
# ==============================
st.header("🤖 ML Model")

df["Env_Code"] = df["Environment"].astype("category").cat.codes

X = df[["Total_ARGs","High_Priority_ARGs","Env_Code"]]
y = (df["ITI_Score"] > 15000).astype(int)

model = LogisticRegression(max_iter=1000)
model.fit(X,y)

arg = st.number_input("ARGs",1000)
hp = st.number_input("High Priority",500)
env = st.selectbox("Env", env_list)

env_code = df[df["Environment"]==env]["Env_Code"].iloc[0]

if st.button("Predict"):
    pred = model.predict([[arg,hp,env_code]])
    st.success("Low Risk") if pred[0]==0 else st.error("High Risk")

# ==============================
# REAL CONFUSION MATRIX
# ==============================
st.header("📊 Model Performance")

y_pred = model.predict(X)
cm = confusion_matrix(y, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

# ==============================
# ROC
# ==============================
rf = RandomForestClassifier()
rf.fit(X,y)

prob = rf.predict_proba(X)[:,1]
fpr,tpr,_ = roc_curve(y,prob)

fig, ax = plt.subplots()
ax.plot(fpr,tpr)
ax.plot([0,1],[0,1],'--')
st.pyplot(fig)

# ==============================
# SHAP (SIMPLIFIED + SAFE)
# ==============================
st.header("🧠 SHAP")

try:
    import shap

    data = pd.read_csv("../data/arg_abundance_matrix.csv")

    Xs = data.drop(columns=["Sample_ID","Environment"])
    ys = data["Environment"]

    model = RandomForestClassifier()
    model.fit(Xs, ys)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xs)

    if isinstance(shap_values, list):
        shap_plot = shap_values[0]
    else:
        shap_plot = shap_values[:,:,0]

    fig = plt.figure()
    shap.summary_plot(shap_plot, Xs, show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP error: {e}")

# ==============================
# BIOMARKERS
# ==============================
st.header("🧬 Biomarkers")

path = "../../ml_system/biomarker_barplot.png"

if os.path.exists(path):
    st.image(path)

# ==============================
# SIMULATION
# ==============================
st.header("📈 Simulation")

sim = "../../ml_system/amr_simulation_results.csv"

if os.path.exists(sim):
    sim_df = pd.read_csv(sim)
    st.dataframe(sim_df)
    st.plotly_chart(px.line(sim_df, x="Growth_Factor", y="Mean_Predicted_ITI"))

# ==============================
# DUAL ML
# ==============================
st.header("🤖 Dual ML")

cm = "../../ml_system/classification_confusion_matrix.png"
reg = "../../ml_system/regression_prediction_plot.png"

if os.path.exists(cm):
    st.image(cm)

if os.path.exists(reg):
    st.image(reg)
