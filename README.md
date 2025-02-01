# Breast Cancer Survival Analysis Using Machine Learning and Statistical Models

## Overview
This project performs survival analysis on a breast cancer dataset using statistical and machine learning models. It implements Kaplan-Meier estimation, Cox Proportional Hazards modeling, and XGBoost classification/regression to analyze survival rates and predict patient outcomes. The project includes preprocessing, model training, evaluation, and comparison.

## Requirements
The following Python libraries are required to run the code:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
lifelines
xgboost
```

Install them using:
```
pip install pandas numpy matplotlib seaborn scikit-learn lifelines xgboost
```

## Dataset
- The dataset is assumed to be a CSV file named `Breast_Cancer2025.csv`.
- It includes categorical and numerical variables such as race, tumor stage, estrogen/progesterone status, and survival months.

## Preprocessing
1. **Handling Missing Values:** Checks for missing data and cleans it if necessary.
2. **Encoding Categorical Features:** Label encoding is applied to categorical variables.
3. **Feature Scaling:** Standardization is applied to numerical features.
4. **Renaming Columns:** Improves consistency and readability.
5. **Dropping Irrelevant Columns:** Removes unimportant features.

## Models Implemented
### 1. **Kaplan-Meier Survival Analysis**
- Estimates the survival function.
- Visualizes survival probability over time.

### 2. **Cox Proportional Hazards Model**
- Evaluates the impact of multiple variables on survival.
- Tests the proportional hazards assumption.

### 3. **XGBoost Model for Classification**
- Predicts survival status (alive or dead) using `XGBClassifier`.
- Evaluates model performance using:
  - Accuracy
  - Confusion matrix
  - ROC Curve and AUC Score

### 4. **XGBoost Regression Model**
- Predicts survival time using `XGBRegressor`.
- Evaluates model using survival predictions.

## Model Comparison
- **Concordance Index (C-index)**
  - Measures predictive performance of Cox and XGBoost models.
- **ROC Curve Analysis**
  - Compares classification model performance visually.

## Visualization
- Kaplan-Meier survival curves.
- Cox survival function.
- ROC curve comparison.
- Bar chart comparing C-indices.

## Usage
1. Run the preprocessing section to clean and prepare the data.
2. Execute each model sequentially.
3. Compare results and interpret survival probabilities and risk factors.

## Expected Output
- Survival probability at specific time points.
- Cox model summary and assumptions check.
- Classification metrics such as accuracy, precision, recall, and AUC.
- Survival time predictions.

## Future Improvements
- Incorporate deep learning models for survival analysis.
- Perform hyperparameter tuning for improved predictions.
- Use external datasets for model validation.

## Author
[Nkudu Uche Victor] - [nkudu4luv@gmail.com]

## License
This project is open-source and available under the MIT License.

