from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_model(data):
    # Prepare data for model
    data_for_model = data.drop(['G1', 'G2'], axis=1)
    X = data_for_model.drop('G3', axis=1)
    y = data_for_model['G3']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate model
    train_score = rf_model.score(X_train, y_train)
    test_score = rf_model.score(X_test, y_test)
    print(f"Training R² Score: {train_score:.2f}")
    print(f"Testing R² Score: {test_score:.2f}")

    # Bin the continuous G3 values into categories for confusion matrix
    bins = [0, 10, 15, 20]  # Bin G3 into "Low" (0-10), "Medium" (10-15), "High" (15-20)
    labels = ['Low', 'Medium', 'High']
    y_test_binned = np.digitize(y_test, bins=bins, right=False)
    y_pred_binned = np.digitize(y_pred, bins=bins, right=False)

    # Adjust for digitize indexing (it starts from 1, we want 0-based indexing for labels)
    y_test_binned = np.clip(y_test_binned - 1, 0, len(labels) - 1)
    y_pred_binned = np.clip(y_pred_binned - 1, 0, len(labels) - 1)

    return rf_model, X.columns, train_score, test_score, y_test_binned, y_pred_binned, labels