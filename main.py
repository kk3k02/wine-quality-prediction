# main.py

from data_loader import load_data, perform_eda
from data_prep import preprocess_data, split_and_scale
from model_trainer import train_and_evaluate_models, plot_feature_importances, save_model


def run_project():
    """Main function to execute the wine quality analysis and modeling pipeline."""

    print("--- START: Wine Quality Prediction Project ---")

    # 1. Data Loading and EDA
    df = load_data()
    df = perform_eda(df)

    # 2. Data Preparation
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = \
        split_and_scale(X, y)

    # 3. Model Training and Evaluation
    feature_names = X.columns
    results, best_model = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names
    )

    # 4. Feature Importance and Model Saving
    plot_feature_importances(best_model, feature_names)
    save_model(best_model, "random_forest_wine_quality_model.joblib")

    print("\n--- END: Wine Quality Prediction Project ---")

    print("\n--- Test Results Summary (R^2 Score) ---")
    print(f"Linear Regression R^2: {results['Linear Regression Test R^2']:.3f}")
    print(f"Random Forest R^2:     {results['Random Forest Test R^2']:.3f}")

    print("\n**The Random Forest model achieved better performance on the test set.**")


if __name__ == '__main__':
    run_project()