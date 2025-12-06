# model_trainer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, \
    r2_score  # Only need mean_squared_error, not the deprecated 'squared' option
import joblib


def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    Trains and evaluates LinearRegression and RandomForestRegressor models.
    """

    print("\n--- Model Training and Evaluation ---")

    # Initialize models
    lin_reg = LinearRegression()
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    models = {'Linear Regression': lin_reg, 'Random Forest': rf_reg}

    results = {}
    best_model = None  # Initialize best_model

    for name, model in models.items():
        print(f"\nModel: {name}")

        # 1. Cross-validation (5-fold, R^2 score)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        print(f"  5-fold CV R^2 scores: {cv_scores.round(3)}")
        results[name + ' CV Mean R^2'] = cv_scores.mean()
        print(f"  CV Mean R^2 = {cv_scores.mean():.3f} (std = {cv_scores.std():.3f})")

        # 2. Train the model on the full training set
        model.fit(X_train_scaled, y_train)

        # 3. Predict on the test set
        y_pred = model.predict(X_test_scaled)

        # 4. Evaluate on the test set - FIX APPLIED HERE
        mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
        rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error (RMSE)
        r2 = r2_score(y_test, y_pred)

        results[name + ' Test RMSE'] = rmse
        results[name + ' Test R^2'] = r2
        print(f"  Test Metrics -> RMSE: {rmse:.3f},  R^2: {r2:.3f}")

        # Store the best model (Random Forest, based on expected performance)
        if name == 'Random Forest':
            best_model = model
            y_pred_rf = y_pred

    # Wizualizacja Actual vs Predicted
    print("\nVisualization: Actual vs Predicted Results")
    y_pred_lin = models['Linear Regression'].predict(X_test_scaled)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred_lin, alpha=0.7, label='Linear Regression')
    # Only scatter RF predictions if best_model was assigned (i.e., RF was run)
    if best_model is not None:
        plt.scatter(y_test, y_pred_rf, alpha=0.7, label='Random Forest')

    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')  # Ideal diagonal line
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title("Actual vs Predicted Quality (Test Set)")
    plt.legend()
    plt.show()

    return results, best_model


def plot_feature_importances(model, feature_names):
    """
    Displays and visualizes feature importances for the RandomForest model.
    """
    if isinstance(model, RandomForestRegressor):
        print("\n--- Feature Importances (Random Forest) ---")
        feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("Random Forest feature importances:")
        print(feature_importances.round(4).to_markdown())

        # Visualize feature importances
        plt.figure(figsize=(6, 4))
        sns.barplot(x=feature_importances, y=feature_importances.index)
        plt.title("Feature Importances (Random Forest)")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        plt.show()
    else:
        print("Feature importances are only available for the Random Forest model.")


def save_model(model, filename="best_wine_quality_model.joblib"):
    """Saves the model using joblib."""
    joblib.dump(model, filename)
    print(f"\nModel saved as {filename}")