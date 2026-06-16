# 💳 Credit Default Prediction

> End-to-end machine learning pipeline for predicting credit card defaults — built on 30,000 records with a focus on recall-optimized risk detection.

---

## 🧠 Project Overview

This project builds a production-style credit risk classification system using a real-world dataset of 30,000 credit card clients. The pipeline covers everything from raw data ingestion to hyperparameter-tuned model evaluation, with explicit prioritization of **recall** to minimize undetected defaults — the metric that matters most in financial risk.

---

## 🚀 Key Highlights

- **Modular sklearn Pipeline** — imputation → encoding → outlier removal → model, all in one composable object
- **Custom OutlierRemover transformer** with configurable σ-band clipping
- **160+ hyperparameter combinations** evaluated via GridSearchCV, RandomizedSearchCV, and HalvingGridSearchCV
- **Recall-first evaluation** — built around credit risk logic, not just accuracy
- **Memory-optimized DataFrame** — downcasting + categorical encoding for production-aware data handling

---

## 📁 Project Structure

```
credit-default-prediction/
│
├── data/
│   └── credit_default.csv          # 30,000-record dataset
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── pipeline.py                 # sklearn Pipeline definition
│   ├── transformers.py             # Custom OutlierRemover transformer
│   └── evaluate.py                 # Metrics and confusion matrix utils
│
├── models/
│   └── best_model.pkl              # Serialized best model
│
├── requirements.txt
└── README.md
```

---

## 🔧 Pipeline Architecture

```
Raw Data
   │
   ▼
Median/Mode Imputation        ← handles missing values
   │
   ▼
OneHot Encoding               ← payment status, education, marital status
   │
   ▼
Custom OutlierRemover         ← configurable σ-band clipping
   │
   ▼
Classifier                    ← Decision Tree / Random Forest
   │
   ▼
Recall-Optimized Predictions
```

---

## 📊 Exploratory Data Analysis

Conducted multivariate EDA to surface credit risk patterns across demographic and behavioral segments:

| Analysis Type | Features Examined |
|---|---|
| Correlation Heatmap | All numeric features |
| Violin Plots | Credit limit by default status |
| KDE Distributions | Age, bill amounts by risk group |
| Default-Rate Bar Charts | Education, marital status, payment behavior |

Key risk signals identified: payment history (PAY_0–PAY_6), credit utilization ratio, and demographic interactions with repayment behavior.

---

## 🤖 Models & Tuning

| Model | Search Strategy | CV Strategy |
|---|---|---|
| Decision Tree | GridSearchCV | Stratified K-Fold |
| Random Forest | RandomizedSearchCV + HalvingGridSearchCV | Stratified K-Fold |

**160+ hyperparameter combinations** tested across:
- `max_depth`, `min_samples_split`, `min_samples_leaf`
- `n_estimators`, `max_features`, `class_weight`

---

## 📈 Evaluation Metrics

Primary metric: **Recall** (minimizing undetected defaults)

| Metric | Description |
|---|---|
| Recall | Default detection rate — primary objective |
| Precision | Signal quality of default flags |
| ROC-AUC | Overall discrimination ability |
| Confusion Matrix | Full breakdown of TP / TN / FP / FN |

> **Why recall?** In credit risk, a missed default (false negative) is far more costly than a false alarm. The pipeline is explicitly optimized to catch as many actual defaults as possible.

---

## ⚙️ Memory Optimization

Applied data governance techniques to reduce DataFrame memory footprint:

- **Dtype downcasting** — `int64 → int32/int16`, `float64 → float32` where precision allows
- **Categorical encoding** — object columns converted to `pd.Categorical`
- Result: significant reduction in memory usage, enabling scalability to larger datasets

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Visualization | `seaborn`, `Plotly` |
| ML & Pipeline | `scikit-learn` |
| Custom Transformers | `sklearn.base.BaseEstimator`, `TransformerMixin` |
| Hyperparameter Tuning | `GridSearchCV`, `RandomizedSearchCV`, `HalvingGridSearchCV` |

---

## 🏃 Getting Started

```bash
# Clone the repo
git clone https://github.com/your-username/credit-default-prediction.git
cd credit-default-prediction

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/pipeline.py
```

---

## 📌 Use Cases & Relevance

This project directly mirrors real-world workflows in:
- **Credit & Fraud Risk** — recall-first modeling, segment-level risk analysis
- **Financial Services Analytics** — pipeline design, feature engineering on behavioral data
- **Data Engineering** — memory optimization, modular, reusable transformer components

---

## 📄 License

MIT License — free to use, adapt, and build on.
