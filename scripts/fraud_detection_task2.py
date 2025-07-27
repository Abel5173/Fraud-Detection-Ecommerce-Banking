import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

# Set up directory paths
BASE_DIR = "fraud_detection_project"
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Step 1: Load Preprocessed Data


def load_processed_data():
    fraud_data = pd.read_csv(os.path.join(
        OUTPUT_DIR, "processed_fraud_data.csv"))
    creditcard_data = pd.read_csv(os.path.join(
        OUTPUT_DIR, "processed_creditcard_data.csv"))
    return fraud_data, creditcard_data

# Step 2: Data Preparation


def prepare_data(df, dataset_name):
    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\nClass distribution after SMOTE in {dataset_name} (training):\n", pd.Series(
        y_train_resampled).value_counts(normalize=True))

    return X_train_resampled, X_test, y_train_resampled, y_test

# Step 3: Train and Evaluate Models


def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    # Initialize models
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }

    results = []

    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train)

        # Predict probabilities and labels
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {model_name} ({dataset_name})")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(
            METRICS_DIR, f"{dataset_name}_{model_name}_confusion_matrix.png"))
        plt.close()

        # Save results
        results.append({
            'model': model_name,
            'AUC-PR': auc_pr,
            'F1-Score': f1
        })

        # Save model
        joblib.dump(model, os.path.join(
            MODEL_DIR, f"{dataset_name}_{model_name}_model.pkl"))

    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(
        METRICS_DIR, f"{dataset_name}_metrics.csv"), index=False)

    return results_df

# Step 4: Justify Model Selection


def justify_model_selection(fraud_results, creditcard_results):
    print("\nModel Performance Comparison:")
    print("\nFraud_Data Metrics:\n", fraud_results)
    print("\ncreditcard Metrics:\n", creditcard_results)

    # Example justification (customize based on results)
    justification = """
    Model Selection Justification:
    - Logistic Regression: Provides interpretable results, suitable for explaining fraud predictions to stakeholders. However, it may underperform on complex patterns due to its linear nature.
    - Random Forest: Captures non-linear relationships and interactions between features, likely performing better on imbalanced data due to its ensemble nature.
    Based on AUC-PR and F1-Score, Random Forest is selected as the best model if it shows higher performance, balancing false positives (to avoid customer inconvenience) and false negatives (to minimize financial loss). Logistic Regression is preferred if interpretability is prioritized for business needs.
    """
    with open(os.path.join(METRICS_DIR, "model_selection_justification.txt"), "w") as f:
        f.write(justification)
    print(justification)

# Main execution


def main():
    # Load preprocessed data
    fraud_data, creditcard_data = load_processed_data()

    # Prepare data
    fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test = prepare_data(fraud_data, "Fraud_Data")
    creditcard_X_train, creditcard_X_test, creditcard_y_train, creditcard_y_test = prepare_data(creditcard_data, "creditcard")

    # Train and evaluate models
    fraud_results = train_and_evaluate(fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test, "fraud_data")
    creditcard_results = train_and_evaluate(creditcard_X_train, creditcard_X_test, creditcard_y_train, creditcard_y_test, "creditcard")

    # Justify model selection
    justify_model_selection(fraud_results, creditcard_results)


if __name__ == "__main__":
    main()
