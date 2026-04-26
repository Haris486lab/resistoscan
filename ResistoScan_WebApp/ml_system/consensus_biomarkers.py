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

    def load_data(self):
        df = pd.read_csv("../data/arg_abundance_matrix.csv")
        df = df.fillna(0)
        df.rename(columns={"Environment": "environment"}, inplace=True)
        features = df.columns.drop(["Sample_ID", "environment"])
        return df, list(features)

    def get_importance(self, name, model, X, y):
        if name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
            return model.feature_importances_
        else:
            perm = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            return perm.importances_mean

    def run(self, df, features):

        X = df[features]
        y = df["environment"]

        X_scaled = self.scaler.fit_transform(X)

        for name, model in self.models.items():
            model.fit(X_scaled, y)
            self.feature_importances[name] = self.get_importance(name, model, X_scaled, y)

        imp_df = pd.DataFrame(self.feature_importances, index=features)

        # Normalize
        imp_df = imp_df.div(imp_df.sum(axis=0), axis=1)

        # Score
        imp_df["Mean"] = imp_df.mean(axis=1)
        imp_df["Std"] = imp_df.std(axis=1)
        imp_df["Score"] = imp_df["Mean"] / (imp_df["Std"] + 0.01)

        imp_df = imp_df.sort_values(by="Score", ascending=False)

        # ==============================
        # ADD MODEL AGREEMENT ✅
        # ==============================
        imp_df["Model_Count"] = (imp_df.iloc[:, :5] > 0).sum(axis=1)

        # ==============================
        # FILTER STRONG CONSENSUS ✅
        # ==============================
        strong = imp_df[imp_df["Model_Count"] >= 3]

        strong.to_csv("strong_consensus_biomarkers.csv")

        print("\n🔥 STRONG CONSENSUS BIOMARKERS:\n")

        for i, (gene, row) in enumerate(strong.head(10).iterrows(), 1):
            print(f"{i}. {gene} | Score: {row['Score']:.4f} | Models: {int(row['Model_Count'])}/5")

        return imp_df

    def visualize(self, df):

        top = df[df["Model_Count"] >= 3].head(10)

        plt.figure(figsize=(12, 7), dpi=200)

        plt.barh(top.index[::-1], top["Score"][::-1], edgecolor='black')

        plt.xlabel("Consensus Score", fontsize=14)
        plt.ylabel("Genes", fontsize=14)
        plt.title("Top Consensus Biomarkers", fontsize=16)

        plt.tight_layout()
        plt.savefig("biomarker_barplot.png", dpi=600)
        plt.close()

        print("✅ biomarker_barplot.png generated")

    def save(self, df):
        df.to_csv("consensus_biomarkers.csv")
        print("✅ consensus_biomarkers.csv saved")


def main():

    print("🚀 Running Biomarker System")

    finder = ConsensusBiomarkerFinder()

    df, features = finder.load_data()
    result = finder.run(df, features)

    finder.visualize(result)
    finder.save(result)

    print("🎉 DONE")


if __name__ == "__main__":
    main()
