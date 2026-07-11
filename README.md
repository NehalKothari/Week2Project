# Week2Project
# Credit Card Fraud Detection using Machine Learning

## Project Overview

Credit card fraud has become one of the major financial crimes in today's digital world. Detecting fraudulent transactions accurately is essential to minimize financial losses and improve customer trust.

This project builds a **Machine Learning-based Credit Card Fraud Detection System** using the popular Kaggle Credit Card Fraud Detection dataset. The project compares multiple classification algorithms and identifies the best-performing model for fraud detection.

---

# Objectives

- Detect fraudulent credit card transactions.
- Handle highly imbalanced data using **SMOTE**.
- Compare the performance of multiple Machine Learning models.
- Evaluate models using various performance metrics.
- Select the best model for deployment.

---

# 📂 Dataset

**Dataset Name:** Credit Card Fraud Detection

**Source:**
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Information

- Total Transactions: **284,807**
- Genuine Transactions: **284,315**
- Fraudulent Transactions: **492**
- Features: **31**
- Target Column: **Class**
  - 0 → Genuine Transaction
  - 1 → Fraudulent Transaction

The dataset is highly imbalanced, making it suitable for learning fraud detection techniques.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib

---

# Machine Learning Workflow

1. Data Loading
2. Data Exploration
3. Missing Value Analysis
4. Duplicate Removal
5. Outlier Analysis
6. Data Visualization
7. Feature Scaling (StandardScaler)
8. Train-Test Split
9. Handling Class Imbalance using SMOTE
10. Model Training
11. Model Evaluation
12. Model Comparison

---

# Models Used

## Logistic Regression

A linear classification algorithm used as the baseline model.

### Advantages

- Simple
- Fast
- Easy to interpret

---

## Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy.

### Advantages

- High Accuracy
- Handles non-linear data
- Reduces overfitting
- Better fraud detection performance

---

# Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

# Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|--------|---------:|----------:|-------:|---------:|
| Logistic Regression | 97.32% | 5.39% | 90.53% | 0.9809 |
| Random Forest | **97.83%** | **49.39%** | **85.26%** | **0.9813** |

### Best Model

 **Random Forest**

Random Forest achieved the best balance between Precision and Recall while maintaining an excellent ROC-AUC score.

---

# Visualizations

The project includes:

- Class Distribution
- ROC Curve Comparison
- Feature Importance
- Confusion Matrix

---

# Project Structure

```
Project2
│
├── creditcard.csv
├── main.py
├── README.md
├── requirements.txt
├── best_model.pkl
├── images
│   ├── data_balanced.png
│   ├── ROC_curve.png
│
└── report
    └── Credit_Card_Fraud_Detection_Report.docx
```


# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Week2Project.git
```

Move into the project folder

```bash
cd Week2Project
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Project

```bash
python main.py
```


# Required Libraries

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
joblib
```


# Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Deep Learning Models
- Real-Time Fraud Detection
- FastAPI Deployment
- Streamlit Dashboard
- Explainable AI (SHAP/LIME)


# References

1. Kaggle Credit Card Fraud Detection Dataset
2. Scikit-learn Documentation
3. Imbalanced-learn Documentation
4. Python Documentation


# Author

**Nehal Kothari**

B.Tech Computer Science (Artificial Intelligence)

Machine Learning | Data Science | Artificial Intelligence Enthusiast


