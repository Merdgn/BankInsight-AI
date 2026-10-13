# 🏦 BankInsight AI
### End-to-End Machine Learning & Bank Marketing Intelligence Platform

**BankInsight AI** is an end-to-end machine learning project that predicts bank customers' likelihood of subscribing to a term deposit using demographic, financial, and historical campaign information.

The project combines exploratory data analysis, leakage-aware preprocessing, classification model development, hyperparameter optimization, explainable AI, and an interactive Streamlit application.

**[🚀 Live Application](https://bankinsight-ai.streamlit.app/)** | **[💻 GitHub Repository](https://github.com/Merdgn/BankInsight-AI)** | **[📊 UCI Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)**

---

## 🎯 Project Objectives

- Analyze customer behavior and historical bank marketing data.
- Predict potential term deposit subscriptions before a new marketing contact.
- Compare machine learning classifiers using stratified cross-validation.
- Handle class imbalance and prevent preprocessing data leakage.
- Optimize classification performance and decision thresholds.
- Explain model behavior using Permutation Importance and SHAP.
- Evaluate customer ranking effectiveness using Lift@K.
- Deploy a working ML-powered prediction application.

## 📊 Dataset

**Source:** UCI Machine Learning Repository — Bank Marketing (`bank-full.csv`)

| Attribute | Description |
|---|---|
| Observations | 45,211 |
| Original input features | 16 |
| Selected modeling features | 11 |
| Target | Term deposit subscription (`yes` / `no`) |
| Positive class | 11.7% |
| Training split | 80% |
| Test split | 20% |

The dataset includes customer demographics, financial indicators, and historical marketing interactions.

### Data Quality & Feature Selection

Data quality checks covered missing values, categorical `unknown` values, duplicate records, invalid values, outlier distributions, and feature availability.

For the pre-contact prediction scenario, five current-campaign attributes were excluded:

`duration`, `campaign`, `contact`, `day`, and `month`.

This feature selection reduces the risk of using information unavailable at the intended prediction time. In particular, call duration is only known after a call has occurred.

Categorical values were encoded with One-Hot Encoding, numerical features were transformed using RobustScaler, and historical contact information was handled through a custom feature engineering transformer.

The full preprocessing workflow is integrated into a Scikit-learn Pipeline.

## 🛠️ Technology Stack

**Programming:** Python

**Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn

**Machine Learning:** Scikit-learn, XGBoost

**Model Evaluation:** Stratified K-Fold Cross-Validation, RandomizedSearchCV, ROC-AUC, Average Precision, F1-score, Lift@K

**Explainable AI:** SHAP, Permutation Importance

**Application:** Streamlit

**Development & Version Control:** Google Colab, GitHub

## 🧠 Machine Learning Model Comparison

Four baseline and classification approaches were evaluated using 5-fold stratified cross-validation.

| Model | Precision | Recall | F1-score | Average Precision |
|---|---:|---:|---:|---:|
| Dummy Classifier | 0.0000 | 0.0000 | 0.0000 | 0.1170 |
| Logistic Regression | 0.2233 | 0.5932 | 0.3244 | 0.3442 |
| Random Forest | 0.3060 | 0.4992 | 0.3793 | 0.3744 |
| XGBoost | 0.2707 | 0.5576 | 0.3644 | 0.3811 |

XGBoost achieved the highest Average Precision among the initial models.

### Hyperparameter Optimization

RandomizedSearchCV was applied to XGBoost and Random Forest, using Average Precision as the optimization objective.

| Optimized Model | Best Search CV AP |
|---|---:|
| XGBoost | **0.3852** |
| Random Forest | 0.3778 |

The optimized XGBoost model was selected as the final candidate.

### Decision Threshold Optimization

Out-of-fold predictions were used to select an F1-oriented decision threshold.

- Default threshold: `0.50`
- Selected threshold: **`0.208`**
- Training OOF F1 at selected threshold: **0.4076**

This threshold was fixed before evaluating the independent test dataset.

## 🏆 Final Model Performance

**Model:** Optimized XGBoost

**Evaluation:** Held-out test dataset (9,043 observations)

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8692 |
| Precision | 0.4340 |
| Recall | 0.3885 |
| F1-score | **0.4100** |
| ROC-AUC | **0.7423** |
| Average Precision | **0.3961** |
| Brier Score | 0.0872 |

### Confusion Matrix

| Actual / Predicted | Not Subscribed | Subscribed |
|---|---:|---:|
| Not Subscribed | 7,449 | 536 |
| Subscribed | 647 | 411 |

The final model identified 411 actual subscribers while producing 536 false-positive predictions. It missed 647 subscribers, indicating that recall remains an important limitation.

## 📈 Business-Oriented Evaluation — Lift@K

The model was evaluated as a customer-ranking system by measuring the observed subscription rate among the highest-scoring customers.

| Top Customers | Subscription Rate | Lift |
|---|---:|---:|
| Top 5% | 58.50% | **5.00×** |
| Top 10% | 44.09% | **3.77×** |
| Top 20% | 30.51% | 2.61× |
| Top 30% | 24.14% | 2.06× |

The top 10% of customers ranked by the model exhibited a 44.09% observed subscription rate, compared with the overall test subscription rate of 11.70%.

These results suggest that the model can support customer prioritization research. However, predictive ranking does not establish the causal benefit of contacting a customer.

## 🔍 Explainable AI — SHAP & Feature Importance

Permutation Importance and SHAP were used to analyze the final model.

Important predictive factors included:

- Previous campaign outcome (`poutcome`)
- Customer age (`age`)
- Days since previous contact (`pdays`)
- Housing loan status (`housing`)
- Average account balance (`balance`)

SHAP analysis was performed on 500 sampled training observations and helped visualize the contribution of transformed features to model predictions.

These explanations describe learned model associations, not causal relationships.

## 🌐 Interactive Streamlit Application

**[Launch BankInsight AI](https://bankinsight-ai.streamlit.app/)**

The deployed application allows users to:

- Enter customer demographic and financial information.
- Specify historical marketing campaign information.
- Validate selected input consistency rules.
- Generate real predictions using the trained XGBoost Pipeline.
- View the model subscription score.
- Compare the score against the selected decision threshold.
- Receive a Higher Priority or Lower Priority classification.
- Review held-out test performance metrics.

The application loads a serialized ML Pipeline containing feature engineering, preprocessing, and classification stages.

## 📁 Project Structure

```text
BankInsight-AI/
├── bankinsight/
│   ├── __init__.py
│   └── feature_engineering.py
├── models/
│   ├── bankinsight_xgboost.joblib
│   └── model_metadata.json
├── notebooks/
│   └── 01_data_understanding.ipynb
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## ⚙️ Installation & Usage

Clone the repository:

```bash
git clone https://github.com/Merdgn/BankInsight-AI.git
cd BankInsight-AI
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Python 3.13 is recommended for compatibility with the deployment configuration. The serialized model was originally exported using XGBoost 3.1.4; the deployed environment uses XGBoost 3.1.3. Cross-version compatibility should be verified for additional environments.

## ⚠️ Limitations & Responsible Use

- The dataset contains historical marketing campaign observations and may not reflect current banking behavior.
- The data does not provide sufficient customer identifiers or complete timestamps for fully customer-independent or forward-looking validation.
- Hyperparameter and threshold selection used the training data; reported cross-validation selection scores may be optimistic.
- Customer subscription predictions should not be interpreted as estimates of the causal impact of marketing contact.
- Performance may change across different customer populations and time periods.
- The application is a research and portfolio demonstration, not a validated production banking decision system.
- Historical customer attributes, including age and marital status, raise fairness considerations that would require assessment before operational use.

## 📚 Dataset Attribution

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository.

https://doi.org/10.24432/C5K306

Dataset license: CC BY 4.0.

## 👩‍💻 Author

**Miray Erdoğan**

Computer Engineering Graduate | Artificial Intelligence & Machine Learning

**GitHub:** https://github.com/Merdgn

---

*BankInsight AI — From raw banking data to interpretable machine learning and interactive customer intelligence.*
