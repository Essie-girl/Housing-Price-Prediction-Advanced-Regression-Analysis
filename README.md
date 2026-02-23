# Housing Price Prediction: Comprehensive Regression Analysis

## Project Overview
This project performs a comprehensive regression analysis on the California Housing dataset to predict median house values. It encompasses an in-depth Exploratory Data Analysis (EDA), advanced feature engineering, comparison of multiple machine learning models, hyperparameter tuning, robust evaluation metrics, detailed error analysis, and visualization of feature importances. The goal is to build a robust and interpretable model for predicting housing prices.


## Introduction
Predicting housing prices is a classic machine learning problem with significant real-world applications. This project demonstrates a systematic approach to tackle this problem, from data understanding and preparation to model building, evaluation, and interpretation. The California Housing dataset, widely used for benchmarking, serves as the basis for this analysis.

## Dataset
The dataset used is the California Housing dataset, available through `sklearn.datasets.fetch_california_housing`. It contains 20,640 samples with 8 features describing various aspects of housing districts in California, along with the target variable `MedHouseVal` (Median House Value).

**Features:**
*   MedInc: Median income in block group
*   HouseAge: Median house age in block group
*   AveRooms: Average number of rooms per household
*   AveBedrms: Average number of bedrooms per household
*   Population: Block group population
*   AveOccup: Average number of household members
*   Latitude: Block group latitude
*   Longitude: Block group longitude
*   MedHouseVal: Median house value for California districts (target variable)

## Exploratory Data Analysis (EDA)
In-depth EDA was conducted to understand the data's characteristics:
*   **Distributions**: Histograms and box plots revealed the distributions of all numerical features. `MedInc` and `MedHouseVal` showed right-skewness, `HouseAge` had a bimodal distribution.
*   **Correlations**: A correlation matrix and scatter plots identified strong positive correlations, notably between `MedInc` and `MedHouseVal`, and between geographical features (`Latitude`, `Longitude`) and the target.
*   **Outliers**: Box plots indicated potential outliers in `MedInc`, `Population`, and `AveOccup`.

## Feature Engineering
New, more informative features were created to enhance model performance:
*   `Rooms_per_Person`: `AveRooms / AveOccup` (Captures spaciousness relative to occupancy).
*   `Bedrooms_per_Room`: `AveBedrms / AveRooms` (Indicates the proportion of bedrooms to total rooms).
These features showed significant correlations with `MedHouseVal`.

## Model Comparison & Hyperparameter Tuning
Three regression models were compared:
1.  **RandomForestRegressor** (tuned)
2.  **Linear Regression**
3.  **GradientBoostingRegressor**

The `RandomForestRegressor` underwent hyperparameter tuning using `RandomizedSearchCV` with 5-fold cross-validation. The optimal parameters found were:
*   `n_estimators`: 50
*   `min_samples_split`: 2
*   `min_samples_leaf`: 1
*   `max_features`: 'sqrt'
*   `max_depth`: 30

## Evaluation Metrics & Cross-Validation
Key regression metrics (MAE, MSE, RMSE, R-squared) were calculated for all models. The **Tuned RandomForestRegressor** consistently outperformed the others:

| Model                  | MAE    | MSE    | RMSE   | R2 Score |
| :--------------------- | :----- | :----- | :----- | :------- |
| **Tuned Random Forest**| 0.3324 | 0.2498 | 0.4998 | 0.8116   |
| Linear Regression      | 0.4803 | 0.4424 | 0.6651 | 0.6663   |
| Gradient Boosting      | 0.3607 | 0.2734 | 0.5229 | 0.7938   |

K-fold cross-validation (5 folds) on the Tuned Random Forest model showed a mean RMSE of **0.5145** with a standard deviation of **0.0125**, confirming its stability and generalization ability.

## Error Analysis
Residual plots (`Predicted - Actual` vs. `Actual`) for the Tuned Random Forest model revealed:
*   Generally random distribution around zero, indicating good model fit.
*   Some heteroscedasticity, particularly for higher actual values, suggesting the model struggles more with predicting higher-priced homes.
*   A tendency to underpredict homes near the maximum `MedHouseVal` (5.0).
*   The distribution of residuals was nearly normal with a slight positive skew.

## Feature Importances
Feature importance analysis from the Tuned RandomForestRegressor highlighted:
*   `MedInc` (Median Income) was the most significant predictor (importance: 0.3148).
*   `Latitude` (0.1200), `Rooms_per_Person` (0.1084), and `Longitude` (0.1032) were also highly influential.

This underscores the critical role of economic factors and geographical location in determining housing prices, as well as the value of the engineered features.

## Summary of Findings & Enhancements
**Key Findings:**
*   Tuned RandomForestRegressor is the best-performing model (R2: 0.8116).
*   Cross-validation confirms model stability.
*   Error analysis indicates challenges with higher-priced homes and slight underprediction tendencies.
*   `MedInc`, `Latitude`, `Rooms_per_Person`, and `Longitude` are the most important features.

