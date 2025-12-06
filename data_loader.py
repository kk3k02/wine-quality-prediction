# data_loader.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path='WineQT.csv'):
    """Loads the data from a CSV file."""
    df = pd.read_csv(file_path)
    return df


def perform_eda(df):
    """Performs initial exploratory data analysis (EDA)."""

    print("--- Exploratory Data Analysis (EDA) ---")

    # Shape and columns
    print("\nDataset shape:", df.shape)
    print("Column names:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head(5).to_markdown(index=False))

    # Basic statistics
    print("\nBasic statistics for numerical columns:")
    print(df.describe().round(2).to_markdown())

    # Distribution of target variable (quality)
    print("\nQuality value counts:")
    quality_counts = df['quality'].value_counts().sort_index()
    print(quality_counts.to_markdown())

    # Visualize quality distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df['quality'], bins=range(df['quality'].min(), df['quality'].max() + 2), edgecolor='black')
    plt.title('Quality Score Distribution')
    plt.xlabel('Wine Quality Score')
    plt.ylabel('Frequency')
    plt.show()

    # Correlation matrix
    print("\nCalculating correlation matrix...")
    # Drop 'Id' if it exists
    df_corr = df.drop('Id', axis=1, errors='ignore')
    corr_matrix = df_corr.corr()

    # Visualize correlation matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Matrix of Features and Quality')
    plt.show()

    print("--- EDA Finished ---")
    return df


if __name__ == '__main__':
    # Example usage
    wine_df = load_data()
    perform_eda(wine_df)