# ===============================
# Credit Card Fraud Detection
# ===============================

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


# 1. Load Dataset
data = pd.read_csv("creditcard.csv")
print("Dataset Loaded\n")

# 2. Features & Target
X = data.drop("Class", axis=1)   # INPUT
y = data["Class"]                # OUTPUT

# 3. Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 6. Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Trained Successfully\n")

# 7. Test MULTIPLE random transactions
print("Random Transaction Predictions:\n")

for i in range(10):
    index = random.randint(0, len(X_test) - 1)

    transaction = X_test[index].reshape(1, -1)
    prediction = model.predict(transaction)[0]

    if prediction == 1:
        print(f"Transaction {i+1}: FRAUD ")
    else:
        print(f"Transaction {i+1}: GENUINE ")


# 8. Model Evaluation
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
