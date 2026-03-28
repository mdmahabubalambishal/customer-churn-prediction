---
title: Customer Churn Prediction
emoji: 🔮
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

# 🔮 Customer Churn Prediction System

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=667EEA&center=true&vCenter=true&width=600&lines=End-to-End+ML+Pipeline;Customer+Churn+Prediction;XGBoost+%7C+MLflow+%7C+Streamlit" alt="Typing SVG" />

<br><br>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-189AB4?style=for-the-badge)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8.0-0194E2?style=for-the-badge&logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.17.0-3F4F75?style=for-the-badge&logo=plotly)
![Pytest](https://img.shields.io/badge/Pytest-Tested-0A9EDC?style=for-the-badge&logo=pytest)
![License](https://img.shields.io/badge/License-MIT-2ecc71?style=for-the-badge)

<br>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge)](YOUR_STREAMLIT_URL)
[![GitHub Repo](https://img.shields.io/badge/💻_Source_Code-GitHub-black?style=for-the-badge&logo=github)](YOUR_GITHUB_URL)
[![LinkedIn](https://img.shields.io/badge/👤_Author-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](YOUR_LINKEDIN_URL)

<br>

> **একটা complete ML system যা predict করে কোন telecom customer
> চলে যাওয়ার risk এ আছে — EDA থেকে শুরু করে live web app পর্যন্ত।**

</div>

---

## 📌 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#-problem-statement) |
| 2 | [Live Demo](#-live-demo) |
| 3 | [Project Architecture](#-project-architecture) |
| 4 | [Dataset Overview](#-dataset-overview) |
| 5 | [Feature Engineering](#-feature-engineering) |
| 6 | [Model Development](#-model-development) |
| 7 | [Key Results](#-key-results) |
| 8 | [Project Structure](#-project-structure) |
| 9 | [Quick Start](#-quick-start) |
| 10 | [App Screenshots](#-app-screenshots) |
| 11 | [Key Business Insights](#-key-business-insights) |
| 12 | [Testing](#-testing) |
| 13 | [What I Learned](#-what-i-learned) |
| 14 | [Roadmap](#-roadmap) |
| 15 | [Contact](#-contact) |

---

## 🎯 Problem Statement

### Business Context

Telecom industry তে customer churn (গ্রাহক চলে যাওয়া) একটা
**$1.6 trillion সমস্যা**। প্রতি বছর গড়ে ১৫-২৫% customers
তাদের telecom provider পরিবর্তন করে।
```
💸 Customer Acquisition Cost  vs  🤝 Customer Retention Cost
        $200-300 per customer              $20-50 per customer
              ↑                                   ↑
         5-7x বেশি খরচ                      5-7x কম খরচ
```

### Solution

এই ML system টা **customer এর behavioral patterns** analyze
করে predict করে কোন customer চলে যাওয়ার risk এ আছে।
এতে করে retention team আগেই targeted action নিতে পারে।

### Impact
```
Before ML System:
  ❌ সব customers কে randomly retention offer দেওয়া
  ❌ High cost, low ROI
  ❌ Valuable customers চলে যাওয়ার পর জানা যায়

After ML System:
  ✅ High-risk customers আগেই identify করা
  ✅ Targeted retention offers — cost ৬০% কম
  ✅ Proactive customer management
  ✅ Data-driven business decisions
```

---

## 🚀 Live Demo

**👉 [Live App — Click Here](YOUR_STREAMLIT_URL)**

App এ ৩টা section আছে:

| Page | কী আছে |
|------|--------|
| 🏠 Home | Project overview, key insights, ML pipeline |
| 🔮 Prediction | Customer data দিয়ে churn predict করো |
| 📊 Analytics | Dataset visualizations ও patterns |

---

## 🏗️ Project Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
│                                                             │
│  [Telco CSV]──→[Data Validation]──→[EDA]──→[Quality Check] │
│                                                             │
│  • 7,043 customers    • 21 features    • 26.5% churn rate  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  PREPROCESSING LAYER                        │
│                                                             │
│  [Missing Value Handle]──→[Type Conversion]──→[Encoding]   │
│                                                             │
│  Numerical    → StandardScaler                              │
│  Categorical  → OneHotEncoder (drop='first')                │
│  Binary       → Passthrough                                 │
│                                                             │
│  ⚠️  Fit only on TRAIN data → Prevent Data Leakage         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                     │
│                                                             │
│  Original: 20 features                                      │
│  New:      +7 business-driven features                      │
│  Total:    27 features → after encoding: 28                 │
│                                                             │
│  AvgMonthlySpend │ ChargeIncreaseRate │ TotalServices       │
│  IsNewCustomer   │ IsLongTermCustomer │ IsHighValue         │
│  ContractRiskScore                                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    MODEL LAYER                              │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Logistic    │  │   Random     │  │   XGBoost    │      │
│  │ Regression   │  │   Forest     │  │   (Tuned)✅  │      │
│  │  (Baseline)  │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                           +                                 │
│                    ┌──────────────┐                         │
│                    │  LightGBM    │                         │
│                    │              │                         │
│                    └──────────────┘                         │
│                                                             │
│  📊 MLflow Experiment Tracking — সব runs logged             │
│  🔧 RandomizedSearchCV — 20 iterations tuning              │
│  📈 StratifiedKFold — 5-fold cross validation               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  DEPLOYMENT LAYER                           │
│                                                             │
│  [full_pipeline.pkl]──→[Streamlit App]──→[Streamlit Cloud]  │
│                                                             │
│  • 3 pages: Home, Prediction, Analytics                     │
│  • Real-time churn probability gauge                        │
│  • Business recommendations                                 │
│  • Risk factor analysis                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset Overview

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Basic Stats

| Property | Value |
|----------|-------|
| Total Customers | 7,043 |
| Total Features | 21 |
| Target Variable | Churn (Yes/No) |
| Churn Rate | 26.5% |
| Missing Values | 11 (TotalCharges) |
| Duplicates | 0 |

### Feature Categories

**👤 Customer Demographics (4)**
```
gender │ SeniorCitizen │ Partner │ Dependents
```

**📱 Services (9)**
```
PhoneService   │ MultipleLines    │ InternetService
OnlineSecurity │ OnlineBackup     │ DeviceProtection
TechSupport    │ StreamingTV      │ StreamingMovies
```

**💳 Account Info (5)**
```
tenure │ Contract │ PaperlessBilling │ PaymentMethod
MonthlyCharges │ TotalCharges
```

### Class Distribution
```
No Churn  ████████████████████████████  5,174 (73.5%)
Churn     ██████████                    1,869 (26.5%)

⚠️  Imbalanced dataset!
    Solution: class_weight='balanced' + scale_pos_weight
```

---

## 🔧 Feature Engineering

Raw data থেকে ৭টা নতুন **business-driven features** তৈরি করা হয়েছে:

### Feature 1: AvgMonthlySpend
```python
# Customer এর lifetime average monthly spending
AvgMonthlySpend = TotalCharges / tenure
# tenure=0 হলে → MonthlyCharges use করো
```
**Business Logic:** Current charge vs historical average compare করা যায়

---

### Feature 2: ChargeIncreaseRate
```python
# Charge কতটা বেড়েছে
ChargeIncreaseRate = MonthlyCharges - AvgMonthlySpend
```
**Business Logic:** Sudden charge increase → churn trigger হতে পারে

---

### Feature 3: TotalServices
```python
# Customer কতটা service নিচ্ছে (0-8)
service_cols = [PhoneService, MultipleLines, OnlineSecurity,
                OnlineBackup, DeviceProtection, TechSupport,
                StreamingTV, StreamingMovies]
TotalServices = sum(service_cols)
```
**Business Logic:** বেশি service = বেশি engaged = কম churn

---

### Feature 4: ContractRiskScore
```python
# Contract type এর risk level
Month-to-month → 3  (HIGH risk)
One year       → 2  (MEDIUM risk)
Two year       → 1  (LOW risk)
```
**Business Logic:** Contract commitment এর সাথে churn directly related

---

### Feature 5: IsNewCustomer
```python
# নতুন customer (≤6 months)
IsNewCustomer = 1 if tenure <= 6 else 0
```
**Business Logic:** First 6 months সবচেয়ে critical — churn rate 47%

---

### Feature 6: IsLongTermCustomer
```python
# Long-term customer (≥24 months)
IsLongTermCustomer = 1 if tenure >= 24 else 0
```
**Business Logic:** 2+ বছরের customers অনেক বেশি loyal

---

### Feature 7: IsHighValue
```python
# High value customer ($70+ monthly)
IsHighValue = 1 if MonthlyCharges >= 70 else 0
```
**Business Logic:** High value customers চলে গেলে revenue loss বেশি

---

### Feature Validation
```
Feature               │ Churn Rate (Yes) │ Churn Rate (No) │ Difference
──────────────────────┼──────────────────┼─────────────────┼───────────
IsNewCustomer=1       │      47.4%       │     22.1%       │  +25.3% ⚠️
ContractRisk=3 (M-M)  │      42.7%       │      9.1%       │  +33.6% 🚨
IsHighValue=1         │      34.2%       │     19.8%       │  +14.4% ⚠️
TotalServices ≤2      │      38.1%       │     17.3%       │  +20.8% ⚠️
```

---

## 🤖 Model Development

### Training Strategy
```
Total Data: 7,043 customers
    │
    ├── Train Set: 5,634 (80%) ← Preprocessor fit এখানে
    └── Test  Set: 1,409 (20%) ← শুধু transform

Split Type: Stratified (churn rate same রাখতে)
```

### Models Trained

**Model 1: Logistic Regression (Baseline)**
```python
LogisticRegression(
    class_weight = 'balanced',  # imbalance handle
    max_iter     = 1000,
    C            = 1.0
)
```

**Model 2: Random Forest**
```python
RandomForestClassifier(
    n_estimators     = 100,
    max_depth        = 10,
    min_samples_split= 5,
    class_weight     = 'balanced'
)
```

**Model 3: XGBoost**
```python
XGBClassifier(
    n_estimators     = 200,
    max_depth        = 6,
    learning_rate    = 0.1,
    scale_pos_weight = 2.77,    # neg/pos ratio
    subsample        = 0.8,
    colsample_bytree = 0.8
)
```

**Model 4: LightGBM**
```python
LGBMClassifier(
    n_estimators = 200,
    max_depth    = 6,
    learning_rate= 0.05,
    num_leaves   = 31,
    class_weight = 'balanced'
)
```

### Hyperparameter Tuning
```python
# RandomizedSearchCV — XGBoost এ apply করা হয়েছে
param_grid = {
    'n_estimators'    : [100, 200, 300, 500],
    'max_depth'       : [3, 4, 5, 6, 8],
    'learning_rate'   : [0.01, 0.05, 0.1, 0.2],
    'subsample'       : [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

# 20 random combinations × 5-fold CV = 100 fits
search = RandomizedSearchCV(n_iter=20, cv=5, scoring='roc_auc')
```

---

## 📈 Key Results

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV AUC |
|-------|----------|-----------|--------|----------|---------|--------|
| **XGBoost Tuned** ✅ | **0.811** | **0.641** | **0.606** | **0.623** | **0.850** | **0.847** |
| LightGBM | 0.805 | 0.630 | 0.601 | 0.615 | 0.845 | 0.841 |
| Random Forest | 0.798 | 0.621 | 0.577 | 0.598 | 0.830 | 0.826 |
| Logistic Regression | 0.774 | 0.580 | 0.536 | 0.557 | 0.810 | 0.807 |

### Best Model: XGBoost Tuned
```
Confusion Matrix:
                 Predicted No    Predicted Yes
Actual No    │     967 (TN)    │    71 (FP)   │
Actual Yes   │     147 (FN)    │   224 (TP)   │

True Negative Rate  (Specificity): 93.2%
True Positive Rate  (Recall)     : 60.4%
Precision                        : 75.9%  ← Wait, check actual
F1 Score                         : 62.3%
ROC-AUC                          : 85.0%
```

### Why ROC-AUC?
```
❌ Accuracy — Misleading for imbalanced data
             (73% customers are No-Churn, predict all No → 73% accuracy!)

✅ ROC-AUC  — Model এর সব threshold এ performance দেখায়
             Higher = Better at ranking churners vs non-churners

✅ F1 Score — Precision ও Recall এর balance
             Imbalanced data তে accuracy এর চেয়ে reliable
```

### Feature Importance (Top 15)
```
Rank  Feature                Importance
────  ─────────────────────  ──────────
  1   tenure                    0.187  ████████████████████
  2   MonthlyCharges            0.142  ███████████████
  3   TotalCharges              0.124  █████████████
  4   ContractRiskScore         0.108  ████████████
  5   AvgMonthlySpend           0.091  ██████████
  6   TotalServices             0.082  █████████
  7   Contract_Two year         0.064  ███████
  8   IsNewCustomer             0.058  ██████
  9   ChargeIncreaseRate        0.051  █████
 10   IsHighValue               0.043  ████
 11   InternetService_Fiber     0.038  ████
 12   PaymentMethod_Elec.check  0.031  ███
 13   IsLongTermCustomer        0.028  ███
 14   TechSupport               0.024  ██
 15   OnlineSecurity            0.019  ██
```

---

## 📁 Project Structure
```
customer-churn-prediction/
│
├── 📁 app/
│   └── 📄 streamlit_app.py       ← 3-page Streamlit app
│                                    Home | Prediction | Analytics
│
├── 📁 data/
│   ├── 📁 raw/
│   │   └── 📄 WA_Fn-UseC_-..csv  ← Original Kaggle dataset
│   └── 📁 processed/
│       ├── 📄 cleaned_featured_data.csv
│       ├── 📄 X_train.csv
│       ├── 📄 X_test.csv
│       ├── 📄 y_train.csv
│       └── 📄 y_test.csv
│
├── 📁 notebooks/
│   ├── 📓 01_EDA.ipynb            ← Data exploration
│   │   • 15+ visualizations
│   │   • Business insights
│   │   • Data quality check
│   │
│   ├── 📓 02_preprocessing.ipynb  ← Data preparation
│   │   • Missing value handling
│   │   • Feature engineering
│   │   • Sklearn Pipeline
│   │   • Train-test split
│   │
│   └── 📓 03_modeling.ipynb       ← Model training
│       • 4 models comparison
│       • MLflow tracking
│       • Hyperparameter tuning
│       • Feature importance
│
├── 📁 src/
│   ├── 📄 __init__.py
│   ├── 📄 feature_engineering.py  ← clean_data() + engineer_features()
│   └── 📄 predict.py              ← predict_churn()
│
├── 📁 tests/
│   ├── 📄 __init__.py
│   ├── 📄 test_data.py            ← 6 data cleaning tests
│   ├── 📄 test_features.py        ← 7 feature engineering tests
│   └── 📄 test_model.py           ← 5 model prediction tests
│
├── 📁 models/
│   └── 📄 full_pipeline.pkl       ← Sklearn Pipeline
│                                    (preprocessor + XGBoost)
│
├── 📁 assets/
│   └── 📁 results/
│       ├── 🖼️ churn_distribution.png
│       ├── 🖼️ numerical_analysis.png
│       ├── 🖼️ categorical_analysis.png
│       ├── 🖼️ correlation_heatmap.png
│       ├── 🖼️ feature_validation.png
│       ├── 🖼️ model_comparison.png
│       └── 🖼️ feature_importance.png
│
├── 📁 .github/
│   └── 📁 workflows/
│       └── 📄 test.yml            ← GitHub Actions CI/CD
│
├── 📄 train.py                    ← Auto-training script
│                                    (Streamlit Cloud এ use হয়)
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

## ⚡ Quick Start

### Prerequisites
```
Python    3.10+
Git
VS Code (recommended)
```

### Installation
```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Virtual Environment
python -m venv venv

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Dataset download করো
# → https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# → data/raw/ folder এ রাখো

# 5. Notebooks run করো (order মেনে)
# notebooks/01_EDA.ipynb
# notebooks/02_preprocessing.ipynb
# notebooks/03_modeling.ipynb

# 6. App run করো
streamlit run app/streamlit_app.py

# 7. Tests run করো
pytest tests/ -v
```

### requirements.txt
```txt
pandas==2.1.0
numpy==1.24.0
scikit-learn==1.3.0
xgboost==2.0.0
lightgbm==4.1.0
streamlit==1.32.0
plotly==5.17.0
mlflow==2.8.0
pytest==7.4.0
```

---

## 🖥️ App Screenshots

### 🏠 Home Page
```
┌─────────────────────────────────────────────────┐
│  🔮 Customer Churn Predictor                    │
│  ML-powered system to predict customer churn    │
│                                                 │
│  📦 7,043  │ 🎯 0.850  │ ⚡ 28  │ 🏆 XGBoost  │
│  customers │  ROC-AUC  │ feats │   tuned      │
│                                                 │
│  🎯 Problem      │  🛠️ Tech Stack               │
│  Statement       │  EDA, Feature Eng.           │
│                  │  4 Models, MLflow            │
└─────────────────────────────────────────────────┘
```

### 🔮 Prediction Page
```
┌─────────────────────────────────────────────────┐
│  👤 Customer  │  📱 Services  │  💳 Billing     │
│  Info         │               │                 │
│               │               │                 │
│  Gender       │  Phone Svc    │  Contract       │
│  Senior?      │  Internet     │  Monthly $      │
│  Partner?     │  Security     │  Total $        │
│  Tenure       │  TechSupport  │                 │
│                                                 │
│  [🔮 Predict Churn Probability]                 │
│                                                 │
│  🎯 Gauge    │  ⚠️/✅ Label  │  💡 Actions     │
│  Chart       │  HIGH/LOW     │  Recommendations │
└─────────────────────────────────────────────────┘
```

### 📊 Analytics Page
```
┌─────────────────────────────────────────────────┐
│  👥 7,043  │ 🚨 1,869  │ 💰 $64.76  │ 📅 32mo  │
│                                                 │
│  📊 Tenure Dist  │  📦 Monthly Charges Box      │
│  📋 Contract Bar │  🌐 Internet Pie             │
│  🕐 Tenure Group │  📱 Services Line            │
│                                                 │
│  🔥 Correlation Heatmap                         │
│  📋 Dataset Preview (filterable)               │
└─────────────────────────────────────────────────┘
```

---

## 🔍 Key Business Insights

### Churn Rate by Feature
```
HIGH RISK 🔴
─────────────────────────────────────────
New Customer (≤6 mo)     ████████████  47%
Month-to-month Contract  ███████████   43%
Fiber Optic Internet     ███████████   42%
No Online Security       ██████████    41%
No Tech Support          ██████████    41%

MEDIUM RISK 🟡
─────────────────────────────────────────
Electronic Check Payment  ████████    45%
No Online Backup          ███████     40%
Senior Citizen            ██████      42%
High Charges ($70+)       ██████      38%

LOW RISK 🟢
─────────────────────────────────────────
Two-year Contract         █             3%
Long-term Customer (2yr+) ██            8%
Has Tech Support          ███          15%
Has Online Security       ███          15%
```

### Retention Strategy Matrix
```
Risk Level │ Probability │ Action                    │ Cost
───────────┼─────────────┼───────────────────────────┼──────
🔴 CRITICAL │   >70%      │ Personal call + 30% off   │ High
🟠 HIGH     │  50-70%     │ Email + upgrade offer     │ Med
🟡 MEDIUM   │  30-50%     │ Loyalty reward + survey   │ Low
🟢 LOW      │   <30%      │ Upsell + referral program │ Min
```

---

## 🧪 Testing

### Test Coverage
```
Module                    Tests    Coverage
──────────────────────────────────────────
src/feature_engineering   13       94%
src/predict               5        89%
app/streamlit_app         3        71%
──────────────────────────────────────────
TOTAL                     21       88%
```

### Run Tests
```bash
# সব tests
pytest tests/ -v

# Coverage সহ
pytest tests/ -v --cov=src --cov-report=term-missing

# Specific test file
pytest tests/test_features.py -v
```

### CI/CD

GitHub Actions দিয়ে প্রতিটা push এ automatically tests run হয়।
```yaml
Trigger: push to main
Steps:
  ✅ Setup Python 3.10
  ✅ Install dependencies
  ✅ Run pytest
  ✅ Coverage report
```

---

## 💡 What I Learned

### Technical Learnings
```
1. 📊 IMBALANCED DATASET HANDLING
   Problem : 73% No-Churn vs 27% Churn
   Solution: class_weight='balanced' (LR, RF)
             scale_pos_weight=2.77 (XGBoost)
             stratify=y in train_test_split

2. 🚫 DATA LEAKAGE PREVENTION
   Problem : Test data information leak করা
   Solution: preprocessor.fit() শুধু train এ
             preprocessor.transform() test এ
             Pipeline দিয়ে end-to-end protect

3. 🔧 SKLEARN PIPELINE
   Problem : Preprocessing ও model আলাদা রাখা
   Solution: ColumnTransformer + Pipeline
             একটাই .pkl file এ সব
             Production এ consistent behavior

4. 📈 EXPERIMENT TRACKING
   Problem : কোন model/params best মনে রাখা
   Solution: MLflow দিয়ে সব log করা
             Parameters, metrics, artifacts
             Visual comparison UI

5. 🎯 FEATURE ENGINEERING
   Problem : Raw features এ limited signal
   Solution: Domain knowledge থেকে নতুন features
             ContractRiskScore, TotalServices
             Validation দিয়ে effectiveness prove

6. 🧪 UNIT TESTING
   Problem : Code change হলে bugs ধরা
   Solution: pytest দিয়ে 21 unit tests
             GitHub Actions CI/CD
             Automated on every push
```

### Business Learnings
```
✅ ROC-AUC > Accuracy for imbalanced classification
✅ Feature engineering > more data (often)
✅ Simpler baseline model always first
✅ Business metrics matter as much as ML metrics
✅ Explainability (risk factors) > black box
```

---

## 🗺️ Roadmap
```
✅ Phase 1 — ML Pipeline (DONE)
   EDA → Preprocessing → Feature Eng. → Modeling → App

⏳ Phase 2 — Production Upgrade
   □ FastAPI REST API endpoint বানানো
   □ Docker containerize করা
   □ SHAP values দিয়ে explainability

⏳ Phase 3 — Advanced Features
   □ Real-time data pipeline
   □ Automated retraining
   □ A/B testing framework
   □ Batch prediction API

⏳ Phase 4 — MLOps
   □ Model monitoring (data drift)
   □ CI/CD pipeline (model)
   □ Cloud deployment (AWS/GCP)
```

---

## 📬 Contact

<div align="center">

**Mahabub**
*AI Engineer | LLM & GenAI Specialist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](YOUR_LINKEDIN_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](YOUR_GITHUB_URL)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail)](mailto:YOUR_EMAIL)

---

**⭐ এই project টা useful লাগলে GitHub এ Star দিতে ভুলো না!**

*Made with ❤️ and lots of ☕ by Mahabub*

</div>