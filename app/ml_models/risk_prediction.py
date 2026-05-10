from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def train_risk_model(data):

    # Create Risk Label
    data["risk_level"] = (
        (data["quantity_on_hand"] < 1000) &
        (data["backlog"] > 0)
    ).astype(int)

    # Features
    X = data[[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
        "lead-time",
        "quantity_on_hand",
        "backlog"
    ]]

    # Target
    y = data["risk_level"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train Model
    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy