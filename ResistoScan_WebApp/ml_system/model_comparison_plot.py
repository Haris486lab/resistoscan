# ============================================
# MODEL COMPARISON PLOT
# ============================================

import matplotlib.pyplot as plt

# ============================================
# STEP 1: DEFINE MODEL RESULTS
# ============================================

models = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVM",
    "KNN",
    "Voting"
]

accuracies = [
    0.967,   # Logistic Regression
    0.933,   # Random Forest
    0.917,   # XGBoost
    0.900,   # SVM
    0.867,   # KNN
    0.833    # Voting Classifier
]

# ============================================
# STEP 2: CREATE PLOT
# ============================================

plt.figure()

plt.bar(models, accuracies)

# Labels
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Machine Learning Model Comparison for AMR Classification")

# Rotate labels for readability
plt.xticks(rotation=30)

# ============================================
# STEP 3: SAVE FIGURE
# ============================================

plt.savefig("model_comparison.png")

plt.show()
