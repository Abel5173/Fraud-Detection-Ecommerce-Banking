import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import os
import warnings
warnings.filterwarnings('ignore')

# Set up directory paths with proper relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Base Directory: {BASE_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

# Load datasets
fraud_data = pd.read_csv(os.path.join(DATA_DIR, "Fraud_Data.csv"))
ip_data = pd.read_csv(os.path.join(DATA_DIR, "IpAddress_to_Country.csv"))
creditcard_data = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))

print(f"Fraud_Data shape: {fraud_data.shape}")
print(f"IP_Data shape: {ip_data.shape}")
print(f"CreditCard_Data shape: {creditcard_data.shape}")

# Initial EDA for class imbalance
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(data=fraud_data, x='class')
plt.title('Fraud_Data Class Distribution')
plt.xlabel('Class (0=Legitimate, 1=Fraud)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(data=creditcard_data, x='Class')
plt.title('CreditCard_Data Class Distribution')
plt.xlabel('Class (0=Legitimate, 1=Fraud)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=300, bbox_inches='tight')
plt.close()

# Display initial data info
print("\n=== INITIAL DATA ANALYSIS ===")
print(f"Fraud_Data class distribution:\n{fraud_data['class'].value_counts(normalize=True)}")
print(f"CreditCard_Data class distribution:\n{creditcard_data['Class'].value_counts(normalize=True)}")

def handle_missing_values(df, dataset_name):
    """Handle missing values with business-appropriate strategies"""
    print(f"\n=== MISSING VALUES ANALYSIS - {dataset_name} ===")
    missing_summary = df.isnull().sum()
    print(f"Missing values:\n{missing_summary[missing_summary > 0]}")
    
    # Handle missing values based on data type and business context
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                # For numerical data, use median for robustness
                df[col].fillna(df[col].median(), inplace=True)
                print(f"Filled missing values in {col} with median: {df[col].median()}")
            else:
                # For categorical data, use mode
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_val}")
    
    print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df

def clean_data(df, dataset_name):
    """Clean data with business context"""
    print(f"\n=== DATA CLEANING - {dataset_name} ===")
    
    # Remove duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - df.shape[0]
    print(f"Removed {removed_duplicates} duplicates")
    
    # Correct data types
    if dataset_name == "Fraud_Data":
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        print("Converted timestamp columns to datetime")
    
    # Handle outliers for numerical columns (business context: fraud detection)
    if dataset_name == "Fraud_Data":
        numeric_cols = ['purchase_value', 'age']
    else:
        numeric_cols = ['Amount']
    
    for col in numeric_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            print(f"Outliers in {col}: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
            
            # Cap outliers instead of removing (preserve fraud patterns)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def perform_eda(df, dataset_name):
    """Enhanced EDA with business focus"""
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_eda")
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\n=== EXPLORATORY DATA ANALYSIS - {dataset_name} ===")
    
    # Class distribution analysis
    if 'class' in df.columns or 'Class' in df.columns:
        target_col = 'class' if 'class' in df.columns else 'Class'
        class_dist = df[target_col].value_counts(normalize=True)
        print(f"Class distribution:\n{class_dist}")
        
        # Business impact analysis
        if dataset_name == "Fraud_Data":
            fraud_rate = class_dist[1] if 1 in class_dist else 0
            print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.2f}%)")
            print(f"Business impact: High imbalance requires specialized handling")
    
    # Numerical features analysis
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ['class', 'Class']:
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True, bins=50)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            
            plt.subplot(1, 2, 2)
            if 'class' in df.columns:
                sns.boxplot(x='class', y=col, data=df)
                plt.title(f"{col} vs Fraud Class")
            elif 'Class' in df.columns:
                sns.boxplot(x='Class', y=col, data=df)
                plt.title(f"{col} vs Fraud Class")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{col}_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['signup_time', 'purchase_time']:
            plt.figure(figsize=(12, 6))
            
            # Value counts
            value_counts = df[col].value_counts().head(10)
            plt.subplot(1, 2, 1)
            value_counts.plot(kind='bar')
            plt.title(f"Top 10 values in {col}")
            plt.xticks(rotation=45)
            
            # Fraud rate by category
            plt.subplot(1, 2, 2)
            if 'class' in df.columns:
                fraud_by_cat = df.groupby(col)['class'].mean().sort_values(ascending=False).head(10)
                fraud_by_cat.plot(kind='bar')
                plt.title(f"Fraud rate by {col}")
                plt.ylabel('Fraud Rate')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{col}_analysis.png"), dpi=300, bbox_inches='tight')
            plt.close()

def merge_ip_data(fraud_data, ip_data):
    """Enhanced geolocation analysis for fraud detection"""
    print("\n=== GEOLOCATION ANALYSIS ===")
    
    # Convert IP addresses to integer for merging
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)
    ip_data['lower_bound_ip_address'] = ip_data['lower_bound_ip_address'].astype(int)
    ip_data['upper_bound_ip_address'] = ip_data['upper_bound_ip_address'].astype(int)
    
    # Merge datasets
    def map_ip_to_country(ip):
        match = ip_data[(ip_data['lower_bound_ip_address'] <= ip) & 
                       (ip_data['upper_bound_ip_address'] >= ip)]
        return match['country'].iloc[0] if not match.empty else 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    
    # Analyze fraud patterns by country
    country_fraud_analysis = fraud_data.groupby('country')['class'].agg(['count', 'mean']).sort_values('mean', ascending=False)
    print("Top 10 countries by fraud rate:")
    print(country_fraud_analysis.head(10))
    
    # Create high-risk country feature
    high_risk_countries = country_fraud_analysis[country_fraud_analysis['mean'] > country_fraud_analysis['mean'].quantile(0.9)].index
    fraud_data['high_risk_country'] = fraud_data['country'].isin(high_risk_countries).astype(int)
    
    return fraud_data

def feature_engineering(fraud_data):
    """Enhanced feature engineering for fraud detection"""
    print("\n=== FEATURE ENGINEERING ===")
    
    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['month'] = fraud_data['purchase_time'].dt.month
    fraud_data['time_since_signup'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600
    
    # Business hours feature (9 AM - 5 PM)
    fraud_data['business_hours'] = ((fraud_data['hour_of_day'] >= 9) & (fraud_data['hour_of_day'] <= 17)).astype(int)
    
    # Weekend feature
    fraud_data['weekend'] = (fraud_data['day_of_week'] >= 5).astype(int)
    
    # Transaction velocity features
    fraud_data['transaction_count'] = fraud_data.groupby('user_id')['purchase_time'].transform('count')
    fraud_data['avg_purchase_value'] = fraud_data.groupby('user_id')['purchase_value'].transform('mean')
    fraud_data['purchase_value_ratio'] = fraud_data['purchase_value'] / fraud_data['avg_purchase_value']
    
    # Device and browser risk features
    device_fraud_rate = fraud_data.groupby('device_id')['class'].mean()
    browser_fraud_rate = fraud_data.groupby('browser')['class'].mean()
    source_fraud_rate = fraud_data.groupby('source')['class'].mean()
    
    fraud_data['device_risk'] = fraud_data['device_id'].map(device_fraud_rate)
    fraud_data['browser_risk'] = fraud_data['browser'].map(browser_fraud_rate)
    fraud_data['source_risk'] = fraud_data['source'].map(source_fraud_rate)
    
    # Fill NaN values with overall fraud rate
    overall_fraud_rate = fraud_data['class'].mean()
    fraud_data['device_risk'].fillna(overall_fraud_rate, inplace=True)
    fraud_data['browser_risk'].fillna(overall_fraud_rate, inplace=True)
    fraud_data['source_risk'].fillna(overall_fraud_rate, inplace=True)
    
    # Age-based risk features
    fraud_data['age_group'] = pd.cut(fraud_data['age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
    age_fraud_rate = fraud_data.groupby('age_group')['class'].mean()
    fraud_data['age_risk'] = fraud_data['age_group'].map(age_fraud_rate)
    
    print(f"Created {len([col for col in fraud_data.columns if col not in ['user_id', 'device_id', 'ip_address', 'signup_time', 'purchase_time', 'class']])} new features")
    
    return fraud_data

def transform_data_fixed(df, dataset_name):
    """Fixed data transformation that properly handles all data types"""
    print(f"\n=== DATA TRANSFORMATION - {dataset_name} ===")
    
    # Handle class imbalance analysis
    if 'class' in df.columns:
        class_dist = df['class'].value_counts(normalize=True)
        print(f"Class distribution: {class_dist.to_dict()}")
        print(f"Imbalance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
    elif 'Class' in df.columns:
        class_dist = df['Class'].value_counts(normalize=True)
        print(f"Class distribution: {class_dist.to_dict()}")
        print(f"Imbalance ratio: {class_dist[0]/class_dist[1]:.2f}:1")
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Remove datetime and identifier columns that can't be used in modeling
    columns_to_drop = ['signup_time', 'purchase_time', 'user_id', 'device_id', 'ip_address']
    df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns])
    
    # Handle categorical variables first
    categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['class', 'Class']]
    
    if len(categorical_cols) > 0:
        print(f"Processing {len(categorical_cols)} categorical columns: {categorical_cols}")
        # Use Label Encoding for categorical variables
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df_processed[col] = label_encoder.fit_transform(df_processed[col].astype(str))
    
    # Ensure all remaining columns are numeric
    for col in df_processed.columns:
        if col not in ['class', 'Class']:
            if df_processed[col].dtype == 'object':
                # Convert to numeric, replacing non-numeric values with NaN
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Handle missing values in processed data (only for numeric columns)
    numeric_cols = df_processed.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['class', 'Class']]
    
    for col in numeric_cols:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in {col} with median: {median_val}")
    
    # Normalization and Scaling
    scaler = StandardScaler()
    target_col = 'class' if 'class' in df_processed.columns else 'Class'
    feature_cols = [col for col in df_processed.columns if col != target_col]
    
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
    
    print(f"Final dataset shape: {df_processed.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Data types: {df_processed.dtypes.value_counts()}")
    
    return df_processed

# Main execution
if __name__ == "__main__":
    print("=== FRAUD DETECTION - TASK 1: DATA ANALYSIS AND PREPROCESSING (FIXED) ===\n")
    
    # Step 1: Handle Missing Values
    fraud_data = handle_missing_values(fraud_data, "Fraud_Data")
    creditcard_data = handle_missing_values(creditcard_data, "CreditCard_Data")
    
    # Step 2: Clean Data
    fraud_data = clean_data(fraud_data, "Fraud_Data")
    creditcard_data = clean_data(creditcard_data, "CreditCard_Data")
    
    # Step 3: Perform EDA
    perform_eda(fraud_data, "Fraud_Data")
    perform_eda(creditcard_data, "CreditCard_Data")
    
    # Step 4: Merge datasets for geolocation analysis
    fraud_data = merge_ip_data(fraud_data, ip_data)
    
    # Step 5: Feature Engineering
    fraud_data = feature_engineering(fraud_data)
    
    # Step 6: Data Transformation (FIXED VERSION)
    fraud_data_processed = transform_data_fixed(fraud_data, "Fraud_Data")
    creditcard_data_processed = transform_data_fixed(creditcard_data, "CreditCard_Data")
    
    # Step 7: Save processed datasets
    fraud_data_processed.to_csv(os.path.join(OUTPUT_DIR, "processed_fraud_data_fixed.csv"), index=False)
    creditcard_data_processed.to_csv(os.path.join(OUTPUT_DIR, "processed_creditcard_data_fixed.csv"), index=False)
    
    print("\n=== TASK 1 COMPLETED (FIXED) ===")
    print(f"Processed datasets saved to: {OUTPUT_DIR}")
    print(f"- Fraud_Data: {fraud_data_processed.shape}")
    print(f"- CreditCard_Data: {creditcard_data_processed.shape}")
    
    # Verify the data types
    print("\n=== VERIFICATION ===")
    print("Fraud_Data data types:")
    print(fraud_data_processed.dtypes.value_counts())
    print("\nCreditCard_Data data types:")
    print(creditcard_data_processed.dtypes.value_counts()) 