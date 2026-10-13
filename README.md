# 🏦 BankInsight AI

### End-to-End Bank Marketing Prediction & Customer Intelligence System

BankInsight AI is an end-to-end machine learning project designed to predict whether a bank customer will subscribe to a term deposit following a marketing campaign.

The project focuses on data-driven campaign optimization, responsible predictive modeling, and interpretable machine learning.

## 🎯 Project Objectives

- Analyze customer characteristics and historical marketing campaign data.
- Identify patterns associated with term deposit subscriptions.
- Develop and compare multiple classification algorithms.
- Evaluate predictive performance using appropriate classification metrics.
- Reduce data leakage and investigate model limitations.
- Interpret predictions and provide actionable marketing insights.
- Develop an interactive prediction application.

## 📊 Dataset

**Source:** UCI Machine Learning Repository — Bank Marketing

**Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing

- 45,211 observations
- 16 input features
- Binary target: term deposit subscription (`yes` / `no`)
- Customer demographics, financial indicators and campaign history

The project uses the `bank-full.csv` dataset.

## 🛠️ Planned Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Google Colab
- GitHub

## 🧠 Planned Machine Learning Models

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

Models will be compared using precision, recall, F1-score, ROC-AUC, PR-AUC and confusion matrices.

## 🔍 Methodology

1. Problem Definition
2. Data Collection and Understanding
3. Exploratory Data Analysis
4. Data Preprocessing
5. Feature Engineering
6. Model Training and Validation
7. Hyperparameter Optimization
8. Performance Evaluation
9. Explainability and Error Analysis
10. Application Development and Final Reporting

## ⚠️ Data Leakage Prevention

The `duration` feature will be excluded from pre-call prediction models because call duration is unknown before a marketing call takes place.

Model evaluation will also consider class imbalance and the limitations of historical campaign data.

## 📌 Project Status

**Phase 1 — Project Initialization**

This repository will be updated progressively as the project advances.

## 📚 Dataset Attribution

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Dataset licensed under CC BY 4.0.

## 👩‍💻 Author

Miray Erdoğan

Computer Engineering Graduate | Machine Learning & Artificial Intelligence
