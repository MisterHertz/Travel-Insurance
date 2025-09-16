# Travel Insurance Claim Prediction

This project is part of a capstone focusing on predicting whether a travel insurance policy will result in a claim.  
It uses machine learning techniques (Logistic Regression, Random Forest, XGBoost, etc.) and resampling strategies to handle class imbalance.

---

## Project Workflow

1. **Business Understanding**  
   - Travel insurance companies face challenges with **low claim rates** and **imbalanced datasets**.  
   - The goal is to **predict potential claims**, reduce financial risk, and improve decision-making.  

2. **Data Understanding & EDA**  
   - Dataset contains **44,328 rows** and multiple customer/travel attributes.  
   - Checked distributions, correlations, and potential outliers.  
   - No missing values in numeric and categorical features.  
   - Observed significant **class imbalance**: only a small portion of policies had claims.

3. **Data Cleaning**  
   - Removed redundant/unnecessary columns.  
   - Verified no duplicates.  
   - Encoded categorical variables appropriately.

4. **Feature Engineering**  
   - Grouped rare categories (e.g., marital status).  
   - Scaled/normalized numeric features.  
   - Created balanced datasets using **Random Oversampling (ROS)** and **SMOTE** for model comparison.

5. **Modeling**  
   - Tested algorithms:  
     - Logistic Regression  
     - Random Forest  
     - XGBoost  
     - K-Nearest Neighbors  
     - Decision Tree  
   - Used **cross-validation** for performance consistency.  
   - Evaluated with metrics: **Accuracy, Precision, Recall, F1, ROC-AUC**.

6. **Evaluation**  
   - Due to strong imbalance, **ROC-AUC** and **Recall** were more informative than Accuracy.  
   - Models tuned with class weights and resampling.

---

## Key Findings

- Models demonstrate **good ranking ability** with **ROC-AUC ≈ 0.82–0.83** (for example, Logistic Regression ROC-AUC ≈ **0.8287** in the notebook).  
- **Test-set accuracy** is around **76.9%**, but **precision is very low (~4.9%)** while **recall is relatively high (~76.3%)**, producing a **low F1 score (~9.1%)**.  
  → The model captures most true claims (high recall) but produces many false positives (very low precision).  
- **Resampling methods (ROS, SMOTE)** produced **very similar test-set metrics**.  
- **Class-weight balancing** slightly increased recall but did not meaningfully improve precision or F1.  
- **Model comparison:** Logistic Regression showed the highest ROC-AUC (~0.829); XGBoost’s ROC-AUC was lower in this run (~0.778).  
- **Business implication:** The current model is useful as a *high-sensitivity* flagging tool (to surface likely claims for human review), but it needs further refinement (threshold tuning, feature engineering, cost-sensitive optimization) to reduce false positives before automated decisions.

---

## Next Steps

- Apply **threshold tuning** to find better precision/recall trade-offs.  
- Explore **cost-sensitive learning** (assign higher penalty to false negatives).  
- Conduct **feature engineering** on travel and customer variables.  
- Deploy model as a **dashboard or API** for business use.

---

## Repository Structure

