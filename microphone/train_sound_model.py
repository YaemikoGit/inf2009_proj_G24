import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load CSV
df = pd.read_csv("noise_log.csv")

# Features and labels
# (rms, peak, variance, dBFS) -> label
X = df[['rms','peak','variance','dBFS']]
y = df['label']

# Split train/test
# 20% testing data, 80% training data 
# the random_state can be any number, it jus fixes the samples being used (if put like 100 , will be ano set of samples)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree with at most 4 decision layers
# looks at my data and  learn the rules on what value is quiet, normal and noisy
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)

# test the model
y_pred = clf.predict(X_test)

# print the performance results 
print(classification_report(y_test, y_pred))

# Quick test
new_sample = pd.DataFrame([[0.05, 0.5, 0.002, -23]], columns=['rms','peak','variance','dBFS'])
print("Predicted:", clf.predict(new_sample)[0])
