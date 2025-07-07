# model/evaluate.py
from sklearn.metrics import classification_report
import joblib
import pandas as pd

# Dummy eval using same data for illustration
df = pd.read_csv("data/raw_reviews.csv")
X = df["review"]
y = df["label"]

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

X_vec = vectorizer.transform(X)
y_pred = model.predict(X_vec)
print(classification_report(y, y_pred))
