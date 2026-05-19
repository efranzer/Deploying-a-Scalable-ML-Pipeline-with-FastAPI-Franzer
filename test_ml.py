import pytest
import pandas as pd
from ml.data import process_data, apply_label
from ml.model import train_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data/census.csv")
train, test = train_test_split(data, test_size=0.2, random_state=8)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb, scaler = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

model = train_model(X_train, y_train)

def test_apply_label():
    """
    Verify that the binary label in a single inference sample is converted into string output
    """
    assert apply_label([1]) == ">50K", "Binary label conversion incorrect for >50K"
    assert apply_label([0]) == "<=50K", "Binary label conversion incorrect for <=50K"


def test_test_set_size():
    """
    Verify that the test set has at least 500 instances
    """
    assert len(test) >= 500, f"Test set is below 500 instances."


def test_if_log_regression():
    """
    Verifies function returns logistic regression model
    """
    assert isinstance(model, LogisticRegression), "The returned model is not a logistic regression."