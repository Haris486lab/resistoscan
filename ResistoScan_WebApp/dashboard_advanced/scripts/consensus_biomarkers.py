"""
CONSENSUS BIOMARKER DISCOVERY (FINAL PROFESSIONAL VERSION)
- Real ARG data
- Multi-model feature importance
- Clear visualization (readable labels)
- High-resolution output
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class ConsensusBiomarkerFinder:

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }

        self.scaler = StandardScaler()
        self.feature_importances = {}

    # ==============================
    # LOAD DATA
    # ==============================
    def load_data(self):
        print("\n📊 Loading REAL ARG abundance data...")

        df = pd.read_csv("../data/arg_abundance_matrix.csv")
        df = df.fillna(0)

        df.rename(columns={"Environment": "environment"}, inplace=True)

        feature_cols = df.columns.drop(["Sample_ID", "environment"])

        return df, list(feature_cols)

    # ==============================
    # FEATURE IMPORTANCE
    # ==============================
    def get_importance(self, name, model, X, y):

        if name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            return model.feature_importances_
        else:
            perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            return perm.importances_mean

    # ==============================
    # TRAIN + CONSENSUS
    # ==============================
    def run(self, df, features):

        print("\n🧬 Finding consensus biomarkers...")

        X = df[features]
        y = df["environment"]

        X_scaled = self.scaler.fit_transform(X)

        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)

            imp = self.get_importance(name, model, X_scaled, y)
            self.feature_importances[name] = imp

        imp_df = pd.DataFrame(self.feature_importances, index=features)

        # Normalize
        imp_df = imp_df.div(imp_df.sum(axis=0), axis=1)

        # Consensus scoring
        imp_df["Mean"] = imp_df.mean(axis=1)
        imp_df["Std"] = imp_df.std(axis=1)
        imp_df["Score"] = imp_df["Mean"] / (imp_df["Std"] + 0.01)

        imp_df = imp_df.sort_values(by="Score", ascending=False)

        return imp_df

    # ==============================
    # VISUALIZATION (FINAL FIXED)
    # ==============================
    def visualize(self, df):

        top = df.head(10)

        plt.figure(figsize=(12, 7), dpi=200)

        plt.barh(
            top.index[::-1],
            top["Score"][::-1],
            edgecolor='black'
        )

        plt.xlabel("Consensus Score", fontsize=14)
        plt.ylabel("Genes", fontsize=14)
        plt.title("Top 10 Consensus Biomarkers", fontsize=16)

        # 🔥 FIX READABILITY
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)

        plt.grid(axis='x', linestyle='--', alpha=0.5)

        plt.tight_layout()

        plt.savefig("biomarker_barplot.png", dpi=600)
        plt.close()

        print("📊 Saved → biomarker_barplot.png")

    # ==============================
    # REPORT
    # ==============================
    def report(self, df):

        print("\n🏆 TOP BIOMARKERS:\n")

        top = df.head(10)

        for i, (gene, row) in enumerate(top.iterrows(), 1):
            print(f"{i}. {gene} | Score: {row['Score']:.4f}")

        df.to_csv("consensus_biomarkers.csv")
        print("\n💾 Saved: consensus_biomarkers.csv")


# ==============================
# MAIN
# ==============================
def main():

    print("\n🚀 CONSENSUS BIOMARKER SYSTEM\n")

    finder = ConsensusBiomarkerFinder()

    df, features = finder.load_data()

    result = finder.run(df, features)

    finder.visualize(result)

    finder.report(result)

    print("\n✅ DONE → BIOMARKERS GENERATED")


if __name__ == "__main__":
    main()
