# Travel Insurance Claim Prediction - Capstone Project

This repository contains my **Data Science Capstone Project** on predicting whether a customer will make a claim on their travel insurance. The project follows a structured machine learning workflow and includes data exploration, preprocessing, modeling, and evaluation.

---

## Project Overview
The goal of this project is to build a **classification model** that can predict travel insurance claims. The dataset is highly imbalanced, with only ~1.5% of the samples belonging to the positive class (claim).  

Main objectives:
- Explore and clean the data.
- Handle class imbalance.
- Train and evaluate multiple ML models.
- Optimize the top-performing models through hyperparameter tuning.
- Provide insights and recommendations.

---

## Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Checked missing values, distributions, and correlations.  
   - Visualized claim distribution (imbalanced).  

2. **Data Preprocessing**  
   - Encoded categorical variables.  
   - Standardized numerical features.  
   - Split into train-test sets with stratification.  

3. **Modeling**  
   Trained 5 baseline models:  
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
   - Support Vector Machine  
   - K-Nearest Neighbors  

4. **Handling Class Imbalance**  
   - Applied **SMOTE** to oversample the minority class.  

5. **Evaluation Metrics**  
   Since recall is most important (catch as many claim cases as possible), we focused on:  
   - Precision  
   - Recall  
   - F1-Score  
   - ROC-AUC  

6. **Model Selection**  
   - Logistic Regression: highest recall (~0.72 after SMOTE).  
   - XGBoost: best precision but lower recall.  
   - SVM: balanced ROC-AUC performance.  
   - **Top Candidates**: Logistic Regression, XGBoost, SVM.  

7. **Hyperparameter Tuning**  
   - Performed tuning on the top 3 models.  
   - Optimized mainly for **recall**.  

---

## Results

| Model                 | Precision | Recall | F1   | ROC-AUC |
|------------------------|-----------|--------|------|---------|
| Logistic Regression    | 0.05      | 0.72   | 0.10 | 0.81    |
| XGBoost                | 0.07      | 0.36   | 0.12 | 0.80    |
| Support Vector Machine | 0.06      | 0.67   | 0.10 | 0.78    |
| Random Forest          | 0.06      | 0.12   | 0.08 | 0.71    |
| K-Nearest Neighbors    | 0.04      | 0.29   | 0.07 | 0.64    |

**Final Model Chosen:** Logistic Regression (highest recall, stable ROC-AUC).  

---
