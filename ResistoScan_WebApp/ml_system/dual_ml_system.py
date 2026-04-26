import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

# ==============================
# LOAD DATA
# ==============================
print("Loading ITI dataset...")

df = pd.read_csv("../data/iti_scores.csv")

# Encode Environment
df["Env_Code"] = df["Environment"].astype("category").cat.codes

# ==============================
# FEATURES
# ==============================
X = df[["Total_ARGs", "High_Priority_ARGs", "Env_Code"]]

# ==============================
# 🔴 CLASSIFICATION MODEL
# ==============================
print("Training Classification Model...")

y_class = (df["ITI_Score"] > 15000).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class,
    test_size=0.3,
    random_state=42,
    stratify=y_class
)

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {acc:.2f}")
# Cross-validation (Classification)
cv_scores = cross_val_score(classifier, X, y_class, cv=5)

print(f"Classification CV Accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("classification_confusion_matrix.png", dpi=600)
plt.close()

print("Saved classification_confusion_matrix.png")

# ==============================
# 🔵 REGRESSION MODEL
# ==============================
print("Training Regression Model...")

y_reg = df["ITI_Score"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg,
    test_size=0.3,
    random_state=42
)

regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

regressor.fit(X_train_r, y_train_r)

y_pred_r = regressor.predict(X_test_r)

r2 = r2_score(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print(f"Regression R² Score: {r2:.2f}")
# Cross-validation (Regression)
cv_scores_reg = cross_val_score(regressor, X, y_reg, cv=5, scoring='r2')

print(f"Regression CV R²: {cv_scores_reg.mean():.2f} (+/- {cv_scores_reg.std():.2f})")
print(f"RMSE: {rmse:.2f}")

# REGRESSION PLOT
plt.figure(figsize=(6,5))

plt.scatter(y_test_r, y_pred_r)
plt.xlabel("Actual ITI")
plt.ylabel("Predicted ITI")

plt.title(f"Regression Model (R² = {r2:.2f})")

plt.plot(
    [y_test_r.min(), y_test_r.max()],
    [y_test_r.min(), y_test_r.max()],
    'r--'
)

plt.tight_layout()
plt.savefig("regression_prediction_plot.png", dpi=600)
plt.close()

print("Saved regression_prediction_plot.png")

# ==============================
# SAVE MODELS
# ==============================
joblib.dump(classifier, "classifier_model.pkl")
joblib.dump(regressor, "regressor_model.pkl")

print("Models saved")

# ==============================
# DONE
# ==============================
print("\nDUAL ML SYSTEM COMPLETED")
