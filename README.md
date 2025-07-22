# 🛡️ Fraud Detection in E-Commerce Banking

**Author**: Abel  
**Affiliation**: Adey Innovations Inc.  
**Date**: July 2025  

---

## 📌 Project Overview

This project explores and builds predictive models to detect **fraudulent transactions** in e-commerce and banking environments. It combines rigorous **exploratory data analysis**, **feature engineering**, and **machine learning** techniques to identify high-risk patterns, all while considering business implications like cost-sensitive evaluation.

The project uses real-world and synthetic datasets and places strong emphasis on **interpretability**, **model fairness**, and **precision-recall tradeoffs** relevant to fraud detection.

---

## 🧾 Dataset Descriptions

### 1. `FraudData.csv`
- Transaction-level e-commerce data
- Fields: `purchase_value`, `age`, `ip_address`, etc.
- Label: `is_fraud` (binary)

### 2. `creditcard.csv`
- High-dimensional dataset with anonymized principal components (`V1–V28`), `Time`, `Amount`, `Class`
- Known for heavy class imbalance

### 3. `IpAddresstoCountry.csv`
- Maps `ip_address` to geographic `country` for geolocation-based fraud profiling

---

## 🛠️ Project Structure

```bash
📁 fraud-detection-project/
│
├── 📊 fraud_detection_analysis.ipynb    # EDA and insights
├── 🧹 data_preprocessing.py             # Cleaning, imputation, duplicate removal
├── 🧠 feature_engineering.py            # Feature creation (time, IP, frequency)
├── 🔍 model_training.py                 # Model building, SMOTE, evaluation
├── 📈 results/                          # Confusion matrices, metric logs, plots
├── 📄 README.md                         # Project overview and usage
