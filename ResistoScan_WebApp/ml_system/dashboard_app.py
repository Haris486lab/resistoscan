import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

st.set_page_config(layout="wide")

st.title("🧬 AMR Surveillance Dashboard (AI + Explainable ML)")

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("../data/iti_scores.csv")

# ==============================
# SIDEBAR FILTER
# ==============================
st.sidebar.header("🔎 Filters")

env_list = df["Environment"].unique()
selected_env = st.sidebar.multiselect(
    "Select Environment",
    env_list,
    default=env_list
)

filtered_df = df[df["Environment"].isin(selected_env)]

# ==============================
# KPI METRICS
# ==============================
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(filtered_df))
col2.metric("Avg ITI Score", round(filtered_df["ITI_Score"].mean(), 2))
col3.metric("Max ARG Count", int(filtered_df["Total_ARGs"].max()))

# ==============================
# VISUALIZATION
# ==============================
st.subheader("📈 ITI Score Distribution")

st.plotly_chart(
    px.bar(filtered_df, x="Sample_ID", y="ITI_Score", color="Environment"),
    use_container_width=True
)

st.subheader("🧪 ARG vs ITI")

st.plotly_chart(
    px.scatter(
        filtered_df,
        x="Total_ARGs",
        y="ITI_Score",
        color="Environment",
        size="High_Priority_ARGs"
    ),
    use_container_width=True
)

# ==============================
# CORRELATION HEATMAP
# ==============================
st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
sns.heatmap(
    filtered_df[["Total_ARGs", "ITI_Score", "High_Priority_ARGs"]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax
)
st.pyplot(fig)

# ==============================
# ML PREDICTION
# ==============================
st.markdown("---")
st.header("🤖 ML Prediction")

df["Env_Code"] = df["Environment"].astype("category").cat.codes

X = df[["Total_ARGs", "High_Priority_ARGs", "Env_Code"]]
y = (df["ITI_Score"] > 15000).astype(int)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

arg_input = st.number_input("Total ARGs", value=1000)
hp_input = st.number_input("High Priority ARGs", value=500)
env_input = st.selectbox("Environment", env_list)

env_code = df["Environment"].astype("category").cat.codes[df["Environment"] == env_input].iloc[0]

if st.button("Predict Risk"):
    pred = model.predict([[arg_input, hp_input, env_code]])
    st.success("🟢 Low Risk") if pred[0] == 0 else st.error("🔴 High Risk")

# ==============================
# ROC CURVE
# ==============================
st.markdown("---")
st.header("📊 Model Evaluation (ROC Curve)")

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X, y)

y_prob = rf.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], '--')
ax.set_title("ROC Curve")
st.pyplot(fig)

# ==============================
# SHAP EXPLAINABILITY
# ==============================
st.markdown("---")
st.header("🧠 SHAP Explainability")

try:
    import shap

    arg_df = pd.read_csv("../data/arg_abundance_matrix.csv")

    X_arg = arg_df.drop(columns=["Sample_ID", "Environment"])
    y_arg = arg_df["Environment"]

    rf_model = RandomForestClassifier(n_estimators=200)
    rf_model.fit(X_arg, y_arg)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_arg)

    st.subheader("SHAP Summary")

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[0], X_arg, show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP not available: {e}")

# ==============================
# BIOMARKERS
# ==============================
st.markdown("---")
st.header("🧬 Consensus Biomarkers")

bar_path = "../../ml_system/biomarker_barplot.png"

if os.path.exists(bar_path):
    st.image(bar_path, width='stretch')
else:
    st.warning("Run consensus_biomarkers.py")

# ==============================
# SIMULATION
# ==============================
st.markdown("---")
st.header("📈 AMR Simulation")

sim_path = "../../ml_system/amr_simulation_results.csv"

if os.path.exists(sim_path):
    sim_df = pd.read_csv(sim_path)

    st.dataframe(sim_df)

    st.plotly_chart(
        px.line(sim_df, x="Growth_Factor", y="Mean_Predicted_ITI"),
        use_container_width=True
    )
else:
    st.warning("Run predictive_simulation.py")

# ==============================
# DUAL ML RESULTS
# ==============================
st.markdown("---")
st.header("🤖 Dual ML Models")

cm_path = "../../ml_system/classification_confusion_matrix.png"
reg_path = "../../ml_system/regression_prediction_plot.png"

if os.path.exists(cm_path):
    st.subheader("📊 Classification Model")
    st.image(cm_path, width='stretch')

if os.path.exists(reg_path):
    st.subheader("📈 Regression Model")
    st.image(reg_path, width='stretch')

# ==============================
# DOWNLOAD
# ==============================
st.markdown("---")
st.header("📥 Download Data")

st.download_button(
    "Download Filtered Data",
    filtered_df.to_csv(index=False),
    "filtered_data.csv",
    "text/csv"
)
