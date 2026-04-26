# ============================================
# VOTING CLASSIFIER FOR AMR PROJECT
# ============================================

import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# ============================================
# STEP 1: LOAD DATA
# ============================================

DATA_PATH = "/home/harry/ResistoScan_WebApp/data/"

df = pd.read_csv(DATA_PATH + "arg_abundance_matrix.csv")
labels = pd.read_csv(DATA_PATH + "environment_labels.csv")

# Merge
df = df.merge(labels, on="Sample_ID")

# Fix duplicate columns (IMPORTANT)
df["Environment"] = df["Environment_y"]
df.drop(columns=["Environment_x", "Environment_y"], inplace=True)

# ============================================
# STEP 2: PREPARE FEATURES
# ============================================

X = df.drop(columns=["Sample_ID", "Environment"])
y = df["Environment"]

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# STEP 3: DEFINE MODELS
# ============================================

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Voting Classifier
voting_model = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='hard'   # majority voting
)

# ============================================
# STEP 4: LOOCV VALIDATION
# ============================================

loo = LeaveOneOut()
y_true = []
y_pred = []

for train_idx, test_idx in loo.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    voting_model.fit(X_train, y_train)
    pred = voting_model.predict(X_test)

    y_true.append(y_test.values[0])
    y_pred.append(pred[0])

# ============================================
# STEP 5: RESULTS
# ============================================

accuracy = accuracy_score(y_true, y_pred)

print("\n🔥 Voting Classifier Accuracy:", accuracy)

# Save result
with open("voting_classifier_results.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")

print("\n✅ Results saved!")
