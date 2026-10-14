# ShopSmart: Purchase Prediction with Decision Trees

Predicting whether an online store visitor will make a purchase, using session behaviour data (12,330 sessions).

**Live dashboard:** (https://piyushsingh909.github.io/ShopSmart_prediction/)

## Problem
ShopSmart could not tell which visitors were likely to buy, which made marketing inefficient. The goal was to build a Decision Tree classifier that predicts `Revenue` (purchase: yes/no). Only about 16% of visitors buy, so the data is imbalanced and **F1 score** is used instead of accuracy. Target: **F1 > 0.55**.

## Approach
1. Exploratory data analysis (target balance, distributions, correlations)
2. Preprocessing: removed duplicates, converted booleans to 0/1, one-hot encoded `VisitorType` and `Month`
3. Stratified 80/20 train-test split
4. Baseline unpruned tree (shows overfitting)
5. Pruning: pre-pruning with GridSearchCV (5-fold, scoring = F1) and post-pruning with `ccp_alpha`
6. Evaluation on the test set

## Results (test set, 2,441 sessions)

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Model 1 (first run) | 0.71 | 0.62 | **0.662** | 0.90 |
| Model 2 (`class_weight='balanced'`) | 0.51 | **0.86** | 0.639 | 0.85 |

Both models beat the 0.55 benchmark. Model 1 has the best F1. Model 2 finds more buyers (recall 0.86) at the cost of more false alarms.

Pruning check: cross-validation F1 = 0.645, test F1 = 0.639, so the tree is not overfitting.
Best parameters: `max_depth=4`, `min_samples_leaf=5`, `min_samples_split=10`, `criterion='gini'`, `class_weight='balanced'`.

## Files
- `index.html`: the results dashboard (open it in any browser)
- `notebook.ipynb`: full analysis and model code (add your notebook here)

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn
