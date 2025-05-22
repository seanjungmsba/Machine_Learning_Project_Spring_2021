# 👩‍💼 Predicting Female Representation on Company Boards

This project leverages a comprehensive machine learning pipeline to predict **female representation (FEMALE_PCT)** on boards of directors across S&P 1500 firms, drawing from demographic, financial, and organizational datasets. By quantifying board diversity and projecting future trends, the goal is to inform ESG strategies, shape accountability, and support data-driven policy decisions.

---

## 🎯 Business Motivation

While gender equity has seen notable progress in many sectors, **women remain significantly underrepresented on corporate boards**. Regulatory pressure (e.g., Nasdaq’s board diversity rule), investor scrutiny, and societal expectations have pushed diversity disclosure into the spotlight.

This project aims to:
- Quantify board-level gender diversity across firms and sectors
- Forecast expected representation to enable **benchmarking and accountability**
- Identify data-driven levers that most influence female participation

---

## 📊 Data Overview

### 📦 Sources
Data were compiled from **Wharton Research Data Services (WRDS)** platforms, spanning multiple firm-level perspectives:
- **BoardEx** – Board composition, interlocks, and gender counts
- **ExecuComp** – Executive-level compensation and gender
- **Compustat** – Financials including leverage, assets, profitability, liquidity

### 🧮 Merged Dataset Snapshot
- **Observations**: 16,747 firm-year records
- **Time Horizon**: 2006–2022
- **Features**: Over 100 variables including:
  - Board demographics
  - CEO and executive attributes
  - Key financial ratios (ROA, ROE, leverage, liquidity)
  - Industry, geographic, and temporal markers

---

## 🧹 Feature Engineering & Preprocessing

### ✨ Engineered Variables
- `FEMALE_PCT_LAST`: Lagged female board representation
- `CEO_FEMALE`: Indicator variable for female CEO
- `EXEC_COUNT`: Executive headcount from ExecuComp
- `AVG_AGE`, `AVG_TENURE`: Aggregated board experience measures
- `PROFIT_MARGIN`, `DEBT_RATIO`, `LIQUIDITY_RATIO`: Derived from Compustat

### 🏷 Categorical Encoding
- One-hot encoded `GICS Industry`, `Headquarter State`, `Year`
- Standardized numerical columns (z-score) to enable fair treatment across distance-based models

### 🧼 Cleaning Steps
- Removed incomplete board records and financial outliers
- Aligned fiscal year-end dates across WRDS datasets
- Imputed missing values using time-aware forward fill and median backfill

---

## 🧠 Modeling Strategy

The modeling pipeline evaluates a variety of algorithms, from simple linear regression to complex ensembles. It emphasizes **regularization, interpretability, and generalization**.

### 🔨 Models Trained

| Model Type | Algorithm | Notes |
|------------|-----------|-------|
| **Linear** | OLS, Ridge, Lasso, ElasticNet | Establish baselines; inspect multicollinearity |
| **Tree-Based** | Decision Tree | Simple, interpretable structure |
| **Ensemble** | Random Forest, Bagging, ExtraTrees | Improve variance reduction |
| **Boosting** | GradientBoosting, AdaBoost | Focus on correcting prior errors |
| **Stacking** | Ridge + SVR + RF (meta-learner) | Leverages multiple weak learners |
| **HistGradientBoosting** | Scikit-learn’s optimized GBM | Fast and memory-efficient |

### ⚖️ Why Ensemble Models?
Ensembles aggregate predictions from multiple learners to reduce overfitting and variance. Each variant has a specific niche:
- **Bagging** (e.g., Random Forest) reduces variance
- **Boosting** (e.g., XGBoost, AdaBoost) reduces bias
- **Stacking** learns optimal blending of base models

---

## 🧪 Evaluation Metrics

- **R² (Coefficient of Determination)** – Proportion of variance explained
- **MAE (Mean Absolute Error)** – Average absolute deviation from true value
- **RMSE (Root Mean Squared Error)** – Penalizes large errors more heavily

Cross-validation was performed using **k-fold (k=5)** for unbiased generalization performance.

---

## ✅ Best Performing Model

### 🔁 Stacking Regressor
A meta-ensemble combining:
- **Base learners**: Ridge Regression, Decision Tree, SVR
- **Meta learner**: Random Forest Regressor

| Metric        | Value   |
|---------------|---------|
| R² (Test Set) | 0.856   |
| MAE           | 0.029   |
| RMSE          | 0.050   |

---

## 🧭 Interpretation & Use Cases

### 🕵️ Feature Insights
Coefficients and feature importances reveal top drivers:
- Prior year representation (`FEMALE_PCT_LAST`)
- Board size and tenure diversity
- Presence of a female CEO
- Firm size and profitability

### 📊 Benchmarking Tool
The model enables **expected vs actual** comparisons across firms:
- Highlight underperformers vs peers
- Generate percentile-based diversity benchmarks
- Track progress over time for accountability

### 🏛 ESG and Regulatory Relevance
The pipeline supports:
- Diversity disclosures for ESG reporting
- Scenario modeling for proxy advisors
- Internal diversity goal-setting and forecasting

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `female_board_prediction_model.py` | Full modeling pipeline (cleaned & documented) |
| `data/data.csv` | Final modeling dataset (post-ETL) |
| `data/result.xlsx` | Model metrics, feature importances, CV results |
| `Presentation.pdf` | Business summary for stakeholders |
| `Writeup.pdf` | Full technical report (methods, assumptions, model diagnostics) |

---

## 📌 Future Work

- Incorporate **XGBoost / LightGBM** for further boosting performance
- Deploy as API for real-time prediction and benchmarking
- Apply SHAP or LIME for richer feature interpretability
- Automate yearly retraining and ingestion from WRDS pipelines

---

## 🧑‍🔬 Tools and Acknowledgements

- **Tools**: Python, Scikit-learn, Pandas, Statsmodels, Seaborn  
- **Data Access**: WRDS via academic institutional license  

Special thanks to Wharton Research Data Services (WRDS) for providing the critical datasets that enabled this research.
