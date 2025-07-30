import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, auc, f1_score, confusion_matrix, 
                           classification_report, roc_auc_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set up directory paths with proper relative paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print(f"Base Directory: {BASE_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

def load_processed_data():
    """Load preprocessed data from Task 1"""
    try:
        fraud_data = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_fraud_data.csv"))
        creditcard_data = pd.read_csv(os.path.join(OUTPUT_DIR, "processed_creditcard_data.csv"))
        print(f"Loaded processed data - Fraud: {fraud_data.shape}, CreditCard: {creditcard_data.shape}")
        return fraud_data, creditcard_data
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}")
        print("Please run Task 1 first to generate processed data.")
        raise

def prepare_data(df, dataset_name):
    """Prepare data with business-appropriate sampling strategies"""
    print(f"\n=== DATA PREPARATION - {dataset_name} ===")
    
    # Separate features and target
    target_col = 'class' if 'class' in df.columns else 'Class'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"Original class distribution:\n{y.value_counts(normalize=True)}")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set class distribution:\n{y_test.value_counts(normalize=True)}")
    
    # Apply SMOTE to training data only
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Training set after SMOTE:\n{pd.Series(y_train_resampled).value_counts(normalize=True)}")
    
    return X_train_resampled, X_test, y_train_resampled, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test, dataset_name):
    """Train and evaluate models with business-focused metrics"""
    print(f"\n=== MODEL TRAINING AND EVALUATION - {dataset_name} ===")
    
    # Initialize models (simplified version without LightGBM and XGBoost)
    models = {
        'logistic_regression': LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            random_state=42, n_estimators=100, class_weight='balanced'
        )
    }
    
    results = []
    
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict probabilities and labels
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Calculate comprehensive metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision_score_val = precision_score(y_test, y_pred)
        recall_score_val = recall_score(y_test, y_pred)
        
        # Business-focused metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Calculate business cost (example: FP cost = $10, FN cost = $100)
        fp_cost = 10  # Cost of false positive (customer inconvenience)
        fn_cost = 100  # Cost of false negative (fraud loss)
        total_business_cost = (fp * fp_cost) + (fn * fn_cost)
        
        # Save results
        results.append({
            'model': model_name,
            'AUC-PR': auc_pr,
            'ROC-AUC': roc_auc,
            'F1-Score': f1,
            'Precision': precision_score_val,
            'Recall': recall_score_val,
            'False_Positive_Rate': false_positive_rate,
            'False_Negative_Rate': false_negative_rate,
            'Business_Cost': total_business_cost,
            'True_Positives': tp,
            'False_Positives': fp,
            'True_Negatives': tn,
            'False_Negatives': fn
        })
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f"Confusion Matrix - {model_name.title()} ({dataset_name})")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(METRICS_DIR, f"{dataset_name}_{model_name}_confusion_matrix.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create PR curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name.title()} (AUC-PR = {auc_pr:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name.title()} ({dataset_name})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(METRICS_DIR, f"{dataset_name}_{model_name}_pr_curve.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{dataset_name}_{model_name}_model.pkl"))
        
        print(f"Model saved: {dataset_name}_{model_name}_model.pkl")
    
    # Create comparison plots
    results_df = pd.DataFrame(results)
    
    # Model comparison plot
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['AUC-PR', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        results_df.plot(x='model', y=metric, kind='bar', ax=plt.gca())
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
    
    # Business cost comparison
    plt.subplot(2, 3, 6)
    results_df.plot(x='model', y='Business_Cost', kind='bar', ax=plt.gca(), color='red')
    plt.title('Business Cost Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Cost ($)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(METRICS_DIR, f"{dataset_name}_model_comparison.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results_df.to_csv(os.path.join(METRICS_DIR, f"{dataset_name}_detailed_metrics.csv"), index=False)
    
    return results_df

def justify_model_selection(fraud_results, creditcard_results):
    """Provide business-justified model selection"""
    print("\n=== MODEL SELECTION JUSTIFICATION ===")
    
    # Combine results for analysis
    fraud_results['dataset'] = 'Fraud_Data'
    creditcard_results['dataset'] = 'CreditCard_Data'
    all_results = pd.concat([fraud_results, creditcard_results], ignore_index=True)
    
    # Find best models for each dataset
    best_fraud_model = fraud_results.loc[fraud_results['AUC-PR'].idxmax()]
    best_creditcard_model = creditcard_results.loc[creditcard_results['AUC-PR'].idxmax()]
    
    justification = f"""
    MODEL SELECTION JUSTIFICATION
    ============================
    
    BUSINESS CONTEXT:
    - False Positives (FP): Customer inconvenience, potential revenue loss
    - False Negatives (FN): Direct financial loss from undetected fraud
    - Cost ratio: FN cost ($100) >> FP cost ($10)
    
    FRAUD_DATA RESULTS:
    - Best model: {best_fraud_model['model']}
    - AUC-PR: {best_fraud_model['AUC-PR']:.4f}
    - F1-Score: {best_fraud_model['F1-Score']:.4f}
    - Business Cost: ${best_fraud_model['Business_Cost']:.2f}
    - False Positive Rate: {best_fraud_model['False_Positive_Rate']:.4f}
    - False Negative Rate: {best_fraud_model['False_Negative_Rate']:.4f}
    
    CREDITCARD_DATA RESULTS:
    - Best model: {best_creditcard_model['model']}
    - AUC-PR: {best_creditcard_model['AUC-PR']:.4f}
    - F1-Score: {best_creditcard_model['F1-Score']:.4f}
    - Business Cost: ${best_creditcard_model['Business_Cost']:.2f}
    - False Positive Rate: {best_creditcard_model['False_Positive_Rate']:.4f}
    - False Negative Rate: {best_creditcard_model['False_Negative_Rate']:.4f}
    
    RECOMMENDATION:
    """
    
    # Add recommendation based on business context
    if best_fraud_model['Business_Cost'] < best_creditcard_model['Business_Cost']:
        justification += f"""
    For Fraud_Data: {best_fraud_model['model'].title()} is recommended because:
    - Lower business cost (${best_fraud_model['Business_Cost']:.2f} vs ${best_creditcard_model['Business_Cost']:.2f})
    - Better balance of precision and recall
    - More suitable for e-commerce fraud detection with temporal features
    """
    else:
        justification += f"""
    For CreditCard_Data: {best_creditcard_model['model'].title()} is recommended because:
    - Lower business cost (${best_creditcard_model['Business_Cost']:.2f} vs ${best_fraud_model['Business_Cost']:.2f})
    - Better performance on high-dimensional PCA features
    - More suitable for banking transaction fraud detection
    """
    
    justification += f"""
    
    GENERAL OBSERVATIONS:
    - Random Forest generally outperforms Logistic Regression on complex patterns
    - AUC-PR is preferred over accuracy for imbalanced fraud detection
    - Business cost analysis helps balance security vs. customer experience
    - Model selection should consider both performance and interpretability needs
    
    NEXT STEPS:
    - Implement the selected models in production
    - Monitor performance over time
    - Use SHAP analysis for model interpretability (Task 3)
    """
    
    # Save justification
    with open(os.path.join(METRICS_DIR, "model_selection_justification.txt"), "w") as f:
        f.write(justification)
    
    print(justification)
    
    # Create summary table
    summary_table = all_results[['dataset', 'model', 'AUC-PR', 'F1-Score', 'Business_Cost']].copy()
    summary_table = summary_table.sort_values(['dataset', 'AUC-PR'], ascending=[True, False])
    
    print("\nSUMMARY TABLE:")
    print(summary_table.to_string(index=False))
    
    return best_fraud_model, best_creditcard_model

def main():
    """Main execution function"""
    print("=== FRAUD DETECTION - TASK 2: MODEL BUILDING AND TRAINING (SIMPLIFIED) ===")
    print("Note: This version uses only Logistic Regression and Random Forest")
    print("For full functionality with LightGBM and XGBoost, install additional packages\n")
    
    try:
        # Load preprocessed data
        fraud_data, creditcard_data = load_processed_data()
        
        # Prepare data
        fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test = prepare_data(fraud_data, "Fraud_Data")
        creditcard_X_train, creditcard_X_test, creditcard_y_train, creditcard_y_test = prepare_data(creditcard_data, "CreditCard_Data")
        
        # Train and evaluate models
        fraud_results = train_and_evaluate(fraud_X_train, fraud_X_test, fraud_y_train, fraud_y_test, "fraud_data")
        creditcard_results = train_and_evaluate(creditcard_X_train, creditcard_X_test, creditcard_y_train, creditcard_y_test, "creditcard")
        
        # Justify model selection
        best_fraud_model, best_creditcard_model = justify_model_selection(fraud_results, creditcard_results)
        
        print("\n=== TASK 2 COMPLETED ===")
        print(f"Models saved to: {MODEL_DIR}")
        print(f"Metrics saved to: {METRICS_DIR}")
        print(f"Best Fraud_Data model: {best_fraud_model['model']}")
        print(f"Best CreditCard_Data model: {best_creditcard_model['model']}")
        
    except Exception as e:
        print(f"Error in Task 2: {e}")
        raise

if __name__ == "__main__":
    main() 