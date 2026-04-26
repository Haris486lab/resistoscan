# ============================================
# FINAL CLEAN SCRIPT (STATISTICS + MERGE)
# ============================================

import pandas as pd
from scipy.stats import ttest_ind, f_oneway

# ============================================
# STEP 1: LOAD DATA
# ============================================

DATA_PATH = "/home/harry/ResistoScan_WebApp/data/"

df = pd.read_csv(DATA_PATH + "arg_abundance_matrix.csv")
labels = pd.read_csv(DATA_PATH + "environment_labels.csv")

# Merge labels
df = df.merge(labels, on="Sample_ID")
# FIX DUPLICATE ENVIRONMENT COLUMNS
df["Environment"] = df["Environment_y"]
df.drop(columns=["Environment_x", "Environment_y"], inplace=True)
print("\n✅ Data Loaded & Merged Successfully")
print(df.head())

# ============================================
# STEP 2: CHECK ENVIRONMENT DISTRIBUTION
# ============================================

print("\n📊 Sample Distribution:")
print(df["Environment"].value_counts())

# ============================================
# STEP 3: CALCULATE TOTAL ARG ABUNDANCE
# ============================================

# Exclude non-gene columns
non_gene_cols = ["Sample_ID", "Environment"]
gene_columns = [col for col in df.columns if col not in non_gene_cols]

df["Total_ARG"] = df[gene_columns].sum(axis=1)

print("\n✅ Total ARG calculated")

# ============================================
# STEP 4: T-TEST (Sewage vs Gut)
# ============================================

sewage = df[df["Environment"] == "Sewage"]["Total_ARG"]
gut = df[df["Environment"] == "Gut"]["Total_ARG"]

# Check sample size
print("\nSample sizes:")
print("Sewage:", len(sewage))
print("Gut:", len(gut))

if len(sewage) > 1 and len(gut) > 1:
    t_stat, p_value = ttest_ind(sewage, gut)
else:
    t_stat, p_value = None, None

print("\n📊 T-test Results (Sewage vs Gut):")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Save results
with open("t_test_results.txt", "w") as f:
    f.write(f"T-statistic: {t_stat}\nP-value: {p_value}")

# ============================================
# STEP 5: ANOVA (ALL ENVIRONMENTS)
# ============================================

groups = []

for env in df["Environment"].unique():
    group = df[df["Environment"] == env]["Total_ARG"]
    if len(group) > 1:
        groups.append(group)

anova_stat, anova_p = f_oneway(*groups)

print("\n📊 ANOVA Results:")
print("F-statistic:", anova_stat)
print("P-value:", anova_p)

# Save results
with open("anova_results.txt", "w") as f:
    f.write(f"F-statistic: {anova_stat}\nP-value: {anova_p}")

# ============================================
# STEP 6: SAVE SUMMARY
# ============================================

with open("statistical_summary.txt", "w") as f:
    f.write("=== STATISTICAL VALIDATION SUMMARY ===\n\n")
    f.write(f"T-test p-value: {p_value}\n")
    f.write(f"ANOVA p-value: {anova_p}\n")

print("\n✅ All results saved successfully!")
