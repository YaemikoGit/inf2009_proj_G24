import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
df = pd.read_csv("noise_log.csv")

# Features
X = df[["rms", "peak", "variance", "dBFS"]]

# Labels
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "noise_classifier.pkl")

print("Model trained and saved!")