import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# Dummy models (replace with trained models later)

clf = LogisticRegression()
reg = LinearRegression()

def predict_environment(df):
    X = df.select_dtypes(include=[np.number])

    if len(X) == 0:
        return 0

    clf.fit(X, np.random.randint(0, 4, len(X)))
    return clf.predict(X[:1])[0]

def predict_iti(df):
    X = df.select_dtypes(include=[np.number])

    if len(X) == 0:
        return 15000

    y = np.random.uniform(15000, 25000, len(X))
    reg.fit(X, y)

    return reg.predict(X[:1])[0]
