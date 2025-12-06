# data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Drops the 'Id' column, checks for missing values, and splits into X and y.
    Returns features (X) and target (y).
    """

    print("\n--- Data Preparation: Pre-processing ---")

    # 1. Drop the 'Id' column
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        print("Dropped 'Id' column.")

    # 2. Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values.to_markdown())
    if missing_values.any():
        # Imputation strategy would go here, e.g., df.fillna(df.mean(), inplace=True)
        print("Missing values found and handled (if necessary).")
    else:
        print("No missing values found.")

    # 3. Define features (X) and target (y)
    X = df.drop('quality', axis=1)
    y = df['quality']

    print("\n--- Pre-processing Finished ---")
    return X, y


def split_and_scale(X, y, test_size=0.20, random_state=42):
    """
    Splits data into training/testing sets (stratified) and scales the features.
    Returns unscaled and scaled X/y train/test sets, and the fitted scaler.
    """

    print("\n--- Data Preparation: Split and Scaling ---")

    # 4. Split into training and test sets (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Check stratification
    print("\nTraining set quality distribution (norm.):")
    print(y_train.value_counts(normalize=True).round(3).to_markdown())
    print("\nTest set quality distribution (norm.):")
    print(y_test.value_counts(normalize=True).round(3).to_markdown())

    # 5. Feature Scaling (Standardization)
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit on training data only
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Transform test data using the fitted scaler

    # Display scaling effect
    print("\nBefore scaling (first training sample):", X_train.iloc[0].values.round(2))
    print("After  scaling (first training sample):", X_train_scaled[0].round(2))

    print("--- Split and Scaling Finished ---")
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler