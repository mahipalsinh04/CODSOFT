# STEP 1: Libraries import
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# STEP 2: Dataset load (Excel ya CSV)
# Excel ke liye:
# data = pd.read_excel("titanic.xlsx")
# CSV ke liye:
data = pd.read_csv("titanicc.csv")

print("Dataset Loaded:")
print(data.head())  # sirf pehli 5 rows dekhne ke liye

# STEP 3: Missing values handle
# Age me missing value = median se fill
data['Age'].fillna(data['Age'].median(), inplace=True)

# Cabin me missing value = 'G' (bottom deck) fill
data['Cabin'].fillna('G', inplace=True)

# Embarked me missing value = most common port
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# STEP 4: Text columns ko number me badalna
le_sex = LabelEncoder()
le_cabin = LabelEncoder()
le_embarked = LabelEncoder()

data["Sex"] = le_sex.fit_transform(data["Sex"])
data["Cabin"] = le_cabin.fit_transform(data["Cabin"])
data["Embarked"] = le_embarked.fit_transform(data["Embarked"])

# STEP 5: Input (X) aur Output (y)
# Hum simple features use kar rahe hain: Pclass, Sex, Age, Fare, Cabin, Embarked
X = data[["Pclass", "Sex", "Age", "Fare", "Cabin", "Embarked"]]
y = data["Survived"]

# STEP 6: Model banana aur train karna
model = LogisticRegression(max_iter=1000)  # max_iter = convergence ke liye
model.fit(X, y)

print("\nModel trained successfully!")

# STEP 7: New passenger prediction
print("\n--- ENTER NEW PASSENGER DETAILS ---")

pclass = int(input("Enter Ticket Class (1/2/3): "))
sex = input("Enter Gender (male/female): ").strip().lower()
age = float(input("Enter Age: "))
fare = float(input("Enter Fare: "))
cabin = input("Enter Cabin (A-G): ").strip().upper()
embarked = input("Enter Port of Embarkment (C/Q/S): ").strip().upper()

# Safety check: Gender
if sex not in le_sex.classes_:
    print(f"Note: Gender '{sex}' training me nahi tha, default 'male' use kiya gaya.")
    sex = "male"
sex = le_sex.transform([sex])[0]

# Safety check: Cabin
if cabin not in le_cabin.classes_:
    print(f"Note: Cabin '{cabin}' ka data training me nahi tha, default 'G' use kiya gaya.")
    cabin = "G"
cabin = le_cabin.transform([cabin])[0]

# Safety check: Embarked
if embarked not in le_embarked.classes_:
    print(f"Note: Port '{embarked}' training me nahi tha, default 'S' use kiya gaya.")
    embarked = "S"
embarked = le_embarked.transform([embarked])[0]

# STEP 8: Prediction
new_passenger = [[pclass, sex, age, fare, cabin, embarked]]
result = model.predict(new_passenger)

print("\n--- RESULT ---")
if result[0] == 1:
    print("Passenger SURVIVED")
else:
    print("Passenger DID NOT SURVIVE")
