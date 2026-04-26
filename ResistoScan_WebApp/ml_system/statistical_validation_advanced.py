import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu

# ==============================
# LOAD DATA
# ==============================
print("\n📊 Loading data...\n")

df = pd.read_csv("../data/arg_abundance_matrix.csv")
df.columns = df.columns.str.strip()

print("✅ Data Loaded Successfully\n")
print(df.head())

print("\n📊 Sample Distribution:")
print(df["Environment"].value_counts())

# ==============================
# FEATURE SETUP
# ==============================

exclude_cols = ["Sample_ID", "Environment"]
gene_columns = [col for col in df.columns if col not in exclude_cols]

# Total ARG
df["Total_ARG"] = df[gene_columns].sum(axis=1)

# ==============================
# GROUPS (FIXED LABELS)
# ==============================

sewage = df[df["Environment"] == "Sewage"]
gut = df[df["Environment"] == "Human_Gut"]
drain = df[df["Environment"] == "Open_Drain"]
crc = df[df["Environment"] == "CRC"]

print("\n📊 Group Sizes:")
print("Sewage:", len(sewage))
print("Gut:", len(gut))
print("Drain:", len(drain))
print("CRC:", len(crc))

# ==============================
# 🔴 T-TEST (BASELINE)
# ==============================

print("\n📊 T-test (Total ARG: Sewage vs Gut)\n")

t_stat, p_val = ttest_ind(sewage["Total_ARG"], gut["Total_ARG"])
print("T-statistic:", t_stat)
print("P-value:", p_val)

# ==============================
# 🔵 ANOVA
# ==============================

print("\n📊 ANOVA (All Environments)\n")

f_stat, p_anova = f_oneway(
    sewage["Total_ARG"],
    gut["Total_ARG"],
    drain["Total_ARG"],
    crc["Total_ARG"]
)

print("F-statistic:", f_stat)
print("P-value:", p_anova)

# ==============================
# 🟢 BIOMARKERS
# ==============================

biomarkers = ["MexR", "MexS", "nalC", "gyrA", "tetR"]
biomarkers = [gene for gene in biomarkers if gene in df.columns]

print("\n🧬 Using biomarkers:", biomarkers)

# ==============================
# 🔥 BIOMARKER SCORE (FIXED)
# ==============================

df["Biomarker_Score"] = df[biomarkers].mean(axis=1)

sewage_score = df[df["Environment"] == "Sewage"]["Biomarker_Score"]
gut_score = df[df["Environment"] == "Human_Gut"]["Biomarker_Score"]

# ==============================
# 🔥 MANN–WHITNEY TEST
# ==============================

print("\n📊 Mann–Whitney U Test (Biomarker Score)\n")

stat, p_mw = mannwhitneyu(sewage_score, gut_score, method="exact")

print("U-statistic:", stat)
print("P-value:", p_mw)

# ==============================
# 🔥 EFFECT SIZE (CLIFF'S DELTA)
# ==============================

def cliffs_delta(x, y):
    n1, n2 = len(x), len(y)
    greater = sum(i > j for i in x for j in y)
    less = sum(i < j for i in x for j in y)
    return (greater - less) / (n1 * n2)

delta = cliffs_delta(sewage_score, gut_score)

print("\n📊 Effect Size (Cliff's Delta):", delta)

# ==============================
# 📊 VISUALIZATION
# ==============================

plt.figure(figsize=(8, 5))

sns.boxplot(
    x="Environment",
    y="Biomarker_Score",
    data=df
)

plt.title("Biomarker Score Distribution Across Environments")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("biomarker_boxplot.png", dpi=300)
plt.close()

print("\n📊 Saved: biomarker_boxplot.png")

# ==============================
# 💾 SAVE RESULTS
# ==============================

summary = pd.DataFrame({
    "Test": ["T-test", "ANOVA", "Mann-Whitney", "Effect Size"],
    "Value": [p_val, p_anova, p_mw, delta]
})

summary.to_csv("final_statistical_summary.csv", index=False)

print("\n💾 Saved: final_statistical_summary.csv")

print("\n🎉 FINAL STATISTICAL VALIDATION COMPLETE")
