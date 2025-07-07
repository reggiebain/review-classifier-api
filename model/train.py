# model/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv("data/raw_reviews.csv")
X = df["review"]
y = df["label"]

# Split
tf = TfidfVectorizer()
X_vec = tf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "model/model.pkl")
joblib.dump(tf, "model/vectorizer.pkl")
