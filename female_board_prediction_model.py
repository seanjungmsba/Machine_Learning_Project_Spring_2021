"""
Full Regression Modeling Suite
================================

This script provides a comprehensive overview and implementation of various ensemble regression techniques using scikit-learn.
Each model section includes:
- What the model does
- How it works
- Why it's useful

It also includes hyperparameter tuning using GridSearchCV and evaluation metrics such as MAE, MSE, R^2, and explained variance.
"""

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Load data
data = pd.read_csv('data.csv')
X = data.drop(columns=['FEMALE_PCT'])
y = data['FEMALE_PCT']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utility to evaluate models
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    print(f"\n{name} Evaluation")
    print("-------------------------")
    print(f"R2 Score (Train): {model.score(X_train, y_train):.4f}")
    print(f"R2 Score (Test): {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Explained Variance: {explained_variance_score(y_test, y_pred):.4f}")

# --- Gradient Boosting Regressor ---
# What: Builds sequential trees where each corrects its predecessor's error.
# How: Uses gradients of the loss function to optimize predictions iteratively.
# Why: Handles non-linear relationships and reduces bias.
from sklearn.ensemble import GradientBoostingRegressor
print("\nTraining Gradient Boosting Regressor...")
gbr = GradientBoostingRegressor()
gbr_param = {'learning_rate': [0.1], 'n_estimators': [100], 'max_depth': [3]}
gbr_grid = GridSearchCV(gbr, gbr_param, scoring='r2', n_jobs=-1, verbose=0)
gbr_grid.fit(X_train, y_train)
evaluate_model("Gradient Boosting", gbr_grid.best_estimator_)

# --- AdaBoost Regressor ---
# What: Combines many weak learners to create a strong regressor.
# How: Focuses more on previous errors in each iteration.
# Why: Simple and often improves performance on noisy datasets.
from sklearn.ensemble import AdaBoostRegressor
print("\nTraining AdaBoost Regressor...")
adaboost = AdaBoostRegressor(random_state=42)
adaboost_param = {'n_estimators': [100], 'learning_rate': [0.1]}
adaboost_grid = GridSearchCV(adaboost, adaboost_param, scoring='r2', n_jobs=-1, verbose=0)
adaboost_grid.fit(X_train, y_train)
evaluate_model("AdaBoost", adaboost_grid.best_estimator_)

# --- Bagging Regressor ---
# What: Trains models on random subsets and averages predictions.
# How: Reduces variance through bootstrap aggregation.
# Why: Great for unstable models to prevent overfitting.
from sklearn.ensemble import BaggingRegressor
print("\nTraining Bagging Regressor...")
bagging = BaggingRegressor(random_state=42)
bagging_param = {'n_estimators': [100], 'max_samples': [0.8], 'max_features': [0.8]}
bagging_grid = GridSearchCV(bagging, bagging_param, scoring='r2', n_jobs=-1, verbose=0)
bagging_grid.fit(X_train, y_train)
evaluate_model("Bagging", bagging_grid.best_estimator_)

# --- Extra Trees Regressor ---
# What: Ensemble of randomized decision trees.
# How: Splits are chosen randomly instead of optimally.
# Why: Extremely fast and reduces variance significantly.
from sklearn.ensemble import ExtraTreesRegressor
print("\nTraining ExtraTrees Regressor...")
et = ExtraTreesRegressor(random_state=42)
et_param = {'n_estimators': [100], 'max_depth': [None]}
et_grid = GridSearchCV(et, et_param, scoring='r2', n_jobs=-1, verbose=0)
et_grid.fit(X_train, y_train)
evaluate_model("Extra Trees", et_grid.best_estimator_)

# --- Random Forest Regressor ---
# What: Ensemble of decision trees with averaging.
# How: Reduces overfitting by decorrelating trees using feature randomness.
# Why: Strong baseline, works well on structured data.
from sklearn.ensemble import RandomForestRegressor
print("\nTraining Random Forest Regressor...")
rf = RandomForestRegressor(random_state=42)
rf_param = {'n_estimators': [100], 'max_depth': [None]}
rf_grid = GridSearchCV(rf, rf_param, scoring='r2', n_jobs=-1, verbose=0)
rf_grid.fit(X_train, y_train)
evaluate_model("Random Forest", rf_grid.best_estimator_)

# --- Stacking Regressor ---
# What: Combines predictions of multiple regressors using a final meta-model.
# How: Base models make predictions, which are then used as inputs to meta-model.
# Why: Leverages strengths of various models together.
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import StackingRegressor
print("\nTraining Stacking Regressor...")
stacking = StackingRegressor(
    estimators=[
        ('ridge', RidgeCV()),
        ('svr', LinearSVR(max_iter=10000))
    ],
    final_estimator=RandomForestRegressor(n_estimators=10, random_state=42)
)
stacking.fit(X_train, y_train)
evaluate_model("Stacking", stacking)

# --- Histogram-based Gradient Boosting Regressor ---
# What: Boosted trees with histogram optimization.
# How: Buckets continuous features into discrete bins to speed up training.
# Why: High efficiency and accuracy on large datasets.
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
print("\nTraining HistGradientBoosting Regressor...")
hgbr = HistGradientBoostingRegressor()
hgbr_param = {'learning_rate': [0.1], 'max_iter': [100]}
hgbr_grid = GridSearchCV(hgbr, hgbr_param, scoring='r2', n_jobs=-1, verbose=0)
hgbr_grid.fit(X_train, y_train)
evaluate_model("HistGradientBoosting", hgbr_grid.best_estimator_)
