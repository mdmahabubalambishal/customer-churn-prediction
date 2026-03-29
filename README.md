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

<div align="center">

# 🔮 Customer Churn Prediction

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1000&color=667EEA&center=true&vCenter=true&width=600&lines=End-to-End+ML+Pipeline;Customer+Churn+Prediction;XGBoost+%7C+MLflow+%7C+Streamlit;Deployed+on+Hugging+Face+Spaces" alt="Typing SVG" />

<br>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-189AB4?style=for-the-badge)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_Hugging_Face-Deployed-FFD21E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-2ecc71?style=for-the-badge)

<br>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Open_App-FF4B4B?style=for-the-badge)](https://huggingface.co/spaces/mahabub-unlocked/customer-churn-prediction)
[![GitHub](https://img.shields.io/badge/💻_Source_Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/mdmahabubalambishal/customer-churn-prediction)
[![LinkedIn](https://img.shields.io/badge/👤_Connect-LinkedIn-0077B5?style=for-the-badge&logo=linkedin)](www.linkedin.com/in/md-mahabub-alam-bishal-097b77286)

<br>

> **একটা complete ML system যা predict করে কোন telecom customer
> চলে যাওয়ার risk এ আছে — EDA থেকে শুরু করে live web app পর্যন্ত।**

</div>

---

## 🎯 Problem Statement

Telecom industry তে customer churn একটা বড় business সমস্যা।
প্রতি বছর গড়ে ১৫-২৫% customers তাদের telecom provider পরিবর্তন করে।
```
💸 New Customer Cost    vs    🤝 Retention Cost
     $200-300/customer              $20-50/customer
          ↑                               ↑
     5-7x বেশি খরচ               5-7x কম খরচ
```

এই ML system টা **customer এর behavioral patterns** analyze করে
predict করে কোন customer চলে যাওয়ার risk এ আছে — যাতে
retention team আগেই targeted action নিতে পারে।

---

## 🚀 Live Demo

**👉 [App Open করো](https://huggingface.co/spaces/mahabub-unlocked/customer-churn-prediction)**

| Page | কী আছে |
|------|--------|
| 🏠 Home | Project overview, key insights |
| 🔮 Prediction | Customer data দিয়ে churn predict |
| 📊 Analytics | Dataset visualizations |

---

## 🏗️ ML Pipeline
```
Raw Data (CSV)
      │
      ▼
┌─────────────────┐
│   EDA Layer     │  15+ visualizations
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │  StandardScaler + OneHotEncoder
│    Layer        │  Missing value handling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │  7 নতুন business-driven features
│  Engineering    │  ContractRiskScore, TotalServices
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Layer    │  LR → RF → XGBoost → LightGBM
│                 │  MLflow Experiment Tracking
│                 │  RandomizedSearchCV Tuning
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deployment     │  Streamlit Web App
│    Layer        │  Hugging Face Spaces ✅
└─────────────────┘
```

---

## 📊 Dataset Overview

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|----------|-------|
| Total Customers | 7,043 |
| Total Features | 21 |
| Target Variable | Churn (Yes/No) |
| Churn Rate | 26.5% |
| Train/Test Split | 80/20 (Stratified) |

---

## 📈 Model Performance

| Model | ROC-AUC | F1 Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| **XGBoost Tuned ✅** | **0.850** | **0.623** | 0.641 | 0.606 |
| LightGBM | 0.845 | 0.615 | 0.630 | 0.601 |
| Random Forest | 0.830 | 0.598 | 0.621 | 0.577 |
| Logistic Regression | 0.810 | 0.557 | 0.580 | 0.536 |

> **কেন ROC-AUC?**
> Dataset imbalanced (26.5% churn) — accuracy misleading।
> ROC-AUC সব threshold এ model এর performance দেখায়।

---

## 🔧 Feature Engineering

Raw data থেকে ৭টা নতুন business-driven features তৈরি —

| Feature | Formula | Business Logic |
|---------|---------|----------------|
| `AvgMonthlySpend` | TotalCharges / tenure | Historical spending |
| `ChargeIncreaseRate` | MonthlyCharges - AvgMonthlySpend | Charge trend |
| `TotalServices` | Sum of all services (0-8) | Engagement level |
| `ContractRiskScore` | M-M=3, 1yr=2, 2yr=1 | Contract commitment |
| `IsNewCustomer` | tenure ≤ 6 | New customer flag |
| `IsLongTermCustomer` | tenure ≥ 24 | Loyalty flag |
| `IsHighValue` | MonthlyCharges ≥ $70 | Revenue importance |

---

## 🔍 Key Business Insights
```
HIGH RISK 🔴
──────────────────────────────────────────
New Customer (≤6 months)    ████████████  47%
Month-to-month Contract     ███████████   43%
Fiber Optic Internet        ███████████   42%
No Online Security          ██████████    41%
No Tech Support             ██████████    41%

LOW RISK 🟢
──────────────────────────────────────────
Two-year Contract           █              3%
Long-term Customer (2yr+)   ██             8%
Has Tech Support            ███           15%
Has Online Security         ███           15%
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10 |
| ML Models | XGBoost, LightGBM, Random Forest, Logistic Regression |
| ML Library | Scikit-learn |
| Experiment Tracking | MLflow |
| Web App | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Deployment | Hugging Face Spaces |
| Testing | Pytest |
| Version Control | Git + GitHub |

---

## 📁 Project Structure
```
customer-churn-prediction/
│
├── 📄 app.py                     ← HF Spaces entry point
├── 📄 train.py                   ← Auto-training script
├── 📄 requirements.txt
├── 📄 README.md
│
├── 📁 app/
│   └── streamlit_app.py          ← Main Streamlit app
│
├── 📁 src/
│   ├── feature_engineering.py    ← clean_data() + engineer_features()
│   └── predict.py                ← predict_churn()
│
├── 📁 data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
└── 📁 tests/
    ├── test_data.py
    ├── test_features.py
    └── test_model.py
```

---

## ⚡ Quick Start
```bash
# 1. Clone
git clone https://github.com/mdmahabubalambishal/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Run App
streamlit run app/streamlit_app.py

# 5. Tests
pytest tests/ -v
```

---

## 💡 What I Learned
```
✅ Imbalanced Dataset
   class_weight='balanced' + scale_pos_weight
   Stratified train-test split

✅ Data Leakage Prevention
   Preprocessor শুধু train data তে fit
   Test data তে শুধু transform

✅ Feature Engineering
   Domain knowledge থেকে 7 নতুন features
   Business logic → ML signal

✅ Experiment Tracking
   MLflow দিয়ে সব runs log
   Parameters, metrics, models track

✅ Production Pipeline
   Sklearn Pipeline → preprocessing + model
   একটা .pkl file এ সব

✅ Testing & CI/CD
   pytest দিয়ে unit tests
   GitHub Actions automation
```

---

## 🗺️ Roadmap
```
✅ Phase 1 — ML Pipeline (DONE)
⏳ Phase 2 — FastAPI REST API
⏳ Phase 3 — Docker + Cloud Deploy
⏳ Phase 4 — Model Monitoring
⏳ Phase 5 — MLOps Pipeline
```

---

## 📬 Contact

<div align="center">

**Mahabub Alam Bishal**
*AI/ML Engineer | LLM & GenAI Specialist*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](www.linkedin.com/in/md-mahabub-alam-bishal-097b77286)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/mdmahabubalambishal)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Follow-FFD21E?style=for-the-badge)](https://huggingface.co/mahabub-unlocked)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail)](mailto:mdmahabubalambishal@gmail.com)

---

⭐ **এই project টা useful লাগলে GitHub এ Star দাও!**

*Made with ❤️ by Mahabub Alam Bishal*

</div>