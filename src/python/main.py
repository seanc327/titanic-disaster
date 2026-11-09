from pathlib import Path

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import pandas as pd

# print(f"pandas ver: {pd.__version__}")
# print(f"scikit-learn ver: {sklearn.__version__}")

# 13
data_dir = Path(__file__).resolve().parent.parent / "data"
train_path = data_dir / "train.csv"

print(f"Loading data from {train_path}")
train = pd.read_csv(train_path)

# print statements for training data
print(f"Training data loaded:")
print(f"Number of rows: {train.shape[0]}")
print(f"Number of columns: {train.shape[1]}")

# 14
# data exploration
print("Data Exploration:")

print("\nFirst five rows:")
print(train.head())

print("\nMissing values before preparation:")
print(train.isna().sum())

print("\nSurvival counts (0 = died, 1 = survived):")
print(train["Survived"].value_counts())

# features
print("\nAdd features:")
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
print("Added FamilySize = SibSp + Parch + 1 (includes the passenger).")

# data adjustment

# Adjust
print("\nData adjustment:")

features = ["Pclass", "Sex", "Age", "FamilySize", "Fare"]
X = train[features].copy()
y = train["Survived"]

print("Selected features:", features)
print("Target: Survived")
print("Excluded PassengerId, Name, Ticket, Cabin, and Embarked.")

X["Sex"] = X["Sex"].map({"male": 0, "female": 1})
print("\nConverted Sex to binary numbers: male = 0, female = 1.")

age_median = X["Age"].median()
missing_ages = X["Age"].isna().sum()
X["Age"] = X["Age"].fillna(age_median)

print(f"Filled {missing_ages} missing Age values with {age_median}.")

print("\nMissing values after preparation:")
print(X.isna().sum())

print("\nFirst five prepared rows:")
print(X.head())

# 15 log regression model to predict survivability on training set
print("\nBuilding logistic regression model to predict survivability on training set:")

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X, y)
print(f"Trained model on {len(X)} passengers, using StandardScalar to scale features and logistic regression with max_iter = 1000, random_state = 42")

# 16 training accuracy
print("\nTraining accuracy:")

train_predictions = model.predict(X)
train_accuracy = accuracy_score(y, train_predictions)

print(f"Correct predictions: {(train_predictions == y).sum()} out of {len(y)}")
print(f"Training accuracy: {train_accuracy:.2%}")

# 17
print("\nTest data predictions:")

test = pd.read_csv(data_dir / "test.csv")
print(f"Number of rows: {test.shape[0]}")
print(f"Number of columns: {test.shape[1]}")

# prepare test data like how we did train
print("\nPreparing test data:")
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
print("Added FamilySize = SibSp + Parch + 1.")

# features
X_test = test[features].copy()
print("Using same features as training:", features)

X_test["Sex"] = X_test["Sex"].map({"male": 0, "female": 1})
print("Converted Sex to binary, where male = 0, female = 1.")

# missing data
missing_test_ages = X_test["Age"].isna().sum()
X_test["Age"] = X_test["Age"].fillna(age_median)
print(f"Filled {missing_test_ages} missing ages with training median: {age_median}.")

fare_median = X["Fare"].median()
missing_test_fares = X_test["Fare"].isna().sum()
X_test["Fare"] = X_test["Fare"].fillna(fare_median)
print(f"Filled {missing_test_fares} missing fares with training median: {fare_median}.")

print("\nMissing values after test preparation:")
print(X_test.isna().sum())

# predictions
test_predictions = model.predict(X_test)
print("\nApplied the trained scaler and logistic regression model.")

predictions = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

print("\nFirst ten predictions (0 = dead, 1 = survived):")
print(predictions.head(10).to_string(index=False))

print(f"\nGenerated predictions for {len(predictions)} passengers.")
print("Prediction counts:")
print(predictions["Survived"].value_counts().sort_index())

# 18
print("\nTest accuracy:")
print("Since there are no survived labels, we cannot calculate the accuracy for test.csv.")
print("Instead, we will measure accuracy by evaluating on a held-out validation set from train.csv.")

# start with features before missing ages were filled
print("\nValidation preparation:")

X_validation_source = train[features].copy()
print("Used the same features, including FamilySize.")


X_validation_source["Sex"] = X_validation_source["Sex"].map(
    {"male": 0, "female": 1}
)
print("Encoded Sex to binary: male = 0, female = 1.")
print("Kept missing ages for imputation after splitting.")

X_fit, X_valid, y_fit, y_valid = train_test_split(
    X_validation_source,
    train["Survived"],
    test_size=0.2,
    random_state=42,
    stratify=train["Survived"]
)

print(f"Split labeled data: {len(X_fit)} training, {len(X_valid)} validation.")
print("Used a stratified 80/20 split with random_state = 42.")

validation_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

validation_model.fit(X_fit, y_fit)
print("Trained a separate logistic regression model. Median imputation and scaling were fitted on the training split only.")

valid_predictions = validation_model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, valid_predictions)

print(f"\nCorrect validation predictions: "
      f"{(valid_predictions == y_valid).sum()} out of {len(y_valid)}")
print(f"Validation accuracy: {valid_accuracy:.2%}")