import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import os


# Set up directory paths
# BASE_DIR = "fraud_detection_project"
DATA_DIR = os.path.join("../", "data")
OUTPUT_DIR = os.path.join("../", "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
fraud_data = pd.read_csv(os.path.join(DATA_DIR, "Fraud_Data.csv"))
ip_data = pd.read_csv(os.path.join(DATA_DIR, "IpAddress_to_Country.csv"))
creditcard_data = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))


sns.countplot(data=fraud_data, x='class')
sns.countplot(data=creditcard_data, x='Class')
sns.histplot(data=fraud_data, x='purchase_value', hue='class', bins=50, log_scale=True)
sns.histplot(data=creditcard_data, x='Amount',hue='Class', bins=50, log_scale=True)

display(fraud_data.head(), ip_data.head(), creditcard_data.head())

# Step 2: Handle Missing Values


def handle_missing_values(df, dataset_name):
    print(f"\nMissing values in {dataset_name}:\n", df.isnull().sum())
    # Impute numerical columns with median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col].fillna(df[col].median(), inplace=True)
    # Impute categorical columns with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print(
        f"\nMissing values after imputation in {dataset_name}:\n", df.isnull().sum())
    return df

def clean_data(df, dataset_name):
    # Remove duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(
        f"\nRemoved {initial_rows - df.shape[0]} duplicates from {dataset_name}")

    # Correct data types
    if dataset_name == "Fraud_Data":
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df


def perform_eda(df, dataset_name):
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_eda")
    os.makedirs(output_path, exist_ok=True)

    # Univariate Analysis
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col} in {dataset_name}")
        # plt.savefig(os.path.join(output_path, f"{col}_distribution.png"))
        plt.show()
        plt.close()

    # Bivariate Analysis (target vs features)
    if 'class' in df.columns:
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            if col != 'class':
                plt.figure(figsize=(8, 6))
                sns.boxplot(x='class', y=col, data=df)
                plt.title(f"{col} vs Class in {dataset_name}")
                # plt.savefig(os.path.join(output_path, f"{col}_vs_class.png"))
                plt.show()
                plt.close()


def merge_ip_data(fraud_data, ip_data):
    # Convert IP addresses to integer for merging
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(
        int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(
        int)

    # Merge datasets
    def map_ip_to_country(ip):
        match = ip_data[(ip_data['lower_bound_ip_address'] <= ip)
                        & (ip_data['upper_bound_ip_address'] >= ip)]
        return match['country'].iloc[0] if not match.empty else 'Unknown'

    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    return fraud_data


def feature_engineering(fraud_data):
    # Transaction frequency and velocity
    fraud_data['transaction_count'] = fraud_data.groupby(
        'user_id')['purchase_time'].transform('count')
    fraud_data['transaction_velocity'] = fraud_data.groupby('user_id')['purchase_time'].transform(
        lambda x: (x.max() - x.min()).total_seconds() / max(1, x.count()))

    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (
        # in hours
        fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    return fraud_data


def transform_data(df, dataset_name):
    if 'class' in df.columns:
        print(f"\nClass distribution in {dataset_name}:\n",
              df['class'].value_counts(normalize=True))
    else:
        print(
            f"\nWarning: 'class' column not found in {dataset_name}. Skipping class distribution analysis.")

    # Normalization and Scaling
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop([
        'class'], errors='ignore')
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Encode Categorical Features (exclude high-cardinality 'device_id')
    categorical_cols = df.select_dtypes(include=['object']).columns
    if 'device_id' in categorical_cols:
        categorical_cols = categorical_cols.drop(
            'device_id')  # Exclude device_id
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(
            encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    return df


# Clean data
fraud_data = clean_data(fraud_data, "Fraud_Data")
creditcard_data = clean_data(creditcard_data, "creditcard")

# Perform EDA
perform_eda(fraud_data, "Fraud_Data")
perform_eda(creditcard_data, "creditcard")

# Merge datasets
fraud_data = merge_ip_data(fraud_data, ip_data)

# Feature engineering
fraud_data = feature_engineering(fraud_data)

# Data transformation
fraud_data = transform_data(fraud_data, "Fraud_Data")
creditcard_data = transform_data(creditcard_data, "creditcard")

# Save processed datasets
fraud_data.to_csv(os.path.join(
    OUTPUT_DIR, "processed_fraud_data.csv"), index=False)
creditcard_data.to_csv(os.path.join(
    OUTPUT_DIR, "processed_creditcard_data.csv"), index=False)


