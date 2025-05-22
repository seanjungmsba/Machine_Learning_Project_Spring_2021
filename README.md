
# 👩‍💼 Predicting Female Representation on Company Boards

This project applies machine learning to predict the **percentage of female representation** on boards of directors across S&P 1500 firms. It incorporates board demographics, financial metrics, and company attributes to make annual predictions.

---

## 🎯 Business Motivation

Despite progress in gender equity, women remain underrepresented on corporate boards. The goal is to use data to:

- Quantify board diversity
- Predict female representation by firm, sector, and year
- Benchmark performance to drive accountability

---

## 📦 Data Sources

The dataset was assembled using records from **Wharton Research Data Services (WRDS)**:
- **BoardEx** (Board network metrics)
- **ExecuComp** (Executive compensation)
- **Compustat** (Financial ratios)

After merging and cleaning, the dataset included:
- 16,747 annual company records
- 100+ features
- Target: `FEMALE_PCT` (percentage of women on board)

---

## 🧹 Feature Engineering

Key engineered features:
- `CEO_FEMALE`: Indicator for female CEO
- `EXEC_COUNT`: Number of executives
- `FEMALE_PCT_LAST`: Female % from previous year
- Financial ratio indicators (debt, profit, liquidity, etc.)
- Industry, Year, and State one-hot encodings

Normalization was applied to support distance-based models like kNN and Neural Networks.

---

## 🧠 Modeling Strategy

### Algorithms Used
- Linear Models (OLS, Ridge, Lasso, ElasticNet)
- Tree-Based Models (Decision Tree, Random Forest)
- Ensemble Models (Bagging, AdaBoost, Extra Trees, Gradient Boosted Trees)
- Meta-ensemble (Stacking Regressor w/ Ridge + Neural Net + Tree)

### Evaluation Metrics
- **R² (Coefficient of Determination)**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**

---

## 📈 Final Model

**Stacking Regressor** using Ridge, Decision Tree, and Neural Net as base learners:
- R²: 0.856
- RMSE: 0.050
- MAE: 0.029

---

## 🔍 Deployment Use Cases

1. **Feature Analysis**: Understand key drivers of gender equity
2. **Benchmarking**: Predict expected female % and flag low-representation firms
3. **Year-over-Year Tracking**: Evaluate change in board diversity by comparing actual vs predicted values

---

## 🗂 Files

- `female_board_prediction_model.py`: Annotated model training and evaluation code
- `data/data.csv`: Final dataset used for modeling
- `data/result.xlsx`: Output metrics and plots
- `Presentation.pdf`: Stakeholder summary
- `Writeup.pdf`: Full technical and strategic documentation
