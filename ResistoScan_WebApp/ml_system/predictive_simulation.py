import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score


class PredictiveSimulation:

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,   # prevent overfitting
            random_state=42
        )

    # ==============================
    # LOAD DATA
    # ==============================
    def load_data(self):
        print("\n📊 Loading ITI data...")

        df = pd.read_csv("../data/iti_scores.csv")
        df = df.fillna(0)

        return df

    # ==============================
    # TRAIN MODEL + VALIDATION
    # ==============================
    def train_model(self, df):

        print("\n🤖 Training prediction model...")

        X = df[["Total_ARGs", "High_Priority_ARGs"]]
        y = df["ITI_Score"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        print(f"✅ Test R² Score: {r2:.2f}")

        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")
        print(f"✅ CV R²: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")

        return X, y

    # ==============================
    # SIMULATION
    # ==============================
    def simulate_growth(self, df):

        print("\n📈 Simulating future scenarios...")

        scenarios = []

        for growth in [0.8, 1.0, 1.2, 1.5]:
            temp = df.copy()

            temp["Total_ARGs"] *= growth
            temp["High_Priority_ARGs"] *= growth

            pred = self.model.predict(temp[["Total_ARGs", "High_Priority_ARGs"]])

            mean_pred = np.mean(pred)

            # Add risk label
            if mean_pred < 15000:
                risk = "Low"
            elif mean_pred < 22000:
                risk = "Moderate"
            else:
                risk = "High"

            scenarios.append({
                "Growth_Factor": growth,
                "Mean_Predicted_ITI": mean_pred,
                "Risk_Level": risk
            })

        return pd.DataFrame(scenarios)

    # ==============================
    # VISUALIZATION
    # ==============================
    def plot_results(self, sim_df):

        plt.figure(figsize=(8, 5))

        plt.plot(
            sim_df["Growth_Factor"],
            sim_df["Mean_Predicted_ITI"],
            marker="o"
        )

        plt.xlabel("ARG Growth Factor")
        plt.ylabel("Predicted ITI Score")
        plt.title("AMR Risk Simulation")
        plt.grid()

        plt.savefig("amr_simulation.png", dpi=300)
        print("📊 Saved: amr_simulation.png")

    # ==============================
    # SAVE RESULTS
    # ==============================
    def save(self, sim_df):

        sim_df.to_csv("amr_simulation_results.csv", index=False)
        print("💾 Saved: amr_simulation_results.csv")


# ==============================
# MAIN
# ==============================
def main():

    print("\n🚀 PREDICTIVE SIMULATION SYSTEM\n")

    sim = PredictiveSimulation()

    df = sim.load_data()

    sim.train_model(df)

    results = sim.simulate_growth(df)

    print("\n📊 Simulation Results:\n")
    print(results)

    sim.plot_results(results)

    sim.save(results)

    print("\n✅ DONE → FUTURE AMR PREDICTION READY")


if __name__ == "__main__":
    main()
