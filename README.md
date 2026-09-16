# Credit Card Fraud Detection Using XGBoost

A machine learning project that detects fraudulent credit card transactions using XGBoost. The project addresses severe class imbalance with SMOTE and evaluates different classification thresholds to balance fraud detection recall and precision.

## Project Overview

Credit card fraud detection is an imbalanced classification problem because fraudulent transactions represent only a very small percentage of all transactions.

This project builds a binary classification model where:

- **0 — Legitimate:** A normal credit card transaction
- **1 — Fraud:** A fraudulent credit card transaction

The goal is to detect as many fraudulent transactions as possible while limiting the number of legitimate transactions incorrectly flagged as fraud.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset created by the Machine Learning Group at Université Libre de Bruxelles.

Dataset characteristics:

- **Transactions:** 284,807
- **Features:** 30
- **Total columns:** 31
- **Legitimate transactions:** 284,315
- **Fraudulent transactions:** 492
- **Fraud percentage:** Approximately 0.173%
- **Missing values:** 0
- **Target variable:** `Class`

The dataset contains:

- `Time` — Seconds elapsed between each transaction and the first transaction
- `V1` to `V28` — Anonymized features produced using PCA
- `Amount` — Transaction amount
- `Class` — Transaction label, where `1` represents fraud and `0` represents a legitimate transaction

## Project Workflow

The notebook performs the following steps:

1. Imports and explores the dataset
2. Examines the class distribution
3. Checks for missing values
4. Separates the features and target
5. Creates stratified training and test sets
6. Applies SMOTE to the training set
7. Trains an XGBoost classifier
8. Generates fraud probabilities
9. Evaluates the model using ROC-AUC
10. Compares different classification thresholds
11. Produces a classification report and confusion matrix
12. Identifies the most important features

## Class Imbalance

The original dataset is highly imbalanced:

| Class | Transactions | Percentage |
|---|---:|---:|
| Legitimate | 284,315 | 99.827% |
| Fraud | 492 | 0.173% |

A model trained directly on this distribution could predict most transactions as legitimate and still achieve misleadingly high accuracy.

To address this problem, the notebook applies **SMOTE** only to the training set.

### Training Distribution Before SMOTE

| Class | Records |
|---|---:|
| Legitimate | 227,451 |
| Fraud | 394 |

### Training Distribution After SMOTE

| Class | Records |
|---|---:|
| Legitimate | 227,451 |
| Fraud | 22,745 |

SMOTE uses a sampling strategy of `0.1`, making the number of synthetic fraud samples equal to approximately 10% of the legitimate training samples.

## Train-Test Split

The dataset is divided using a stratified split:

- **Training records:** 227,845
- **Test records:** 56,962
- **Test size:** 20%
- **Random state:** 42

Stratification preserves the original fraud ratio in both sets.

## Model

The project uses an XGBoost classifier with the following configuration:

```python
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
```

## Model Results

The model achieved a **ROC-AUC score of 0.9785**.

At the selected classification threshold of `0.50`, the results were:

| Metric | Result |
|---|---:|
| Precision | 0.7265 |
| Recall | 0.8673 |
| F1-score | 0.7907 |
| ROC-AUC | 0.9785 |

## Confusion Matrix

At the selected threshold of `0.50`:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,832 | 32 |
| Actual Fraud | 13 | 85 |

This means:

- **True negatives:** 56,832
- **False positives:** 32
- **False negatives:** 13
- **True positives:** 85

The model correctly detected 85 of the 98 fraudulent transactions in the test set and missed 13 fraudulent transactions.

## Threshold Comparison

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.50 | 0.7265 | 0.8673 | 0.7907 |
| 0.30 | 0.5541 | 0.8878 | 0.6824 |
| 0.20 | 0.4372 | 0.8878 | 0.5859 |
| 0.10 | 0.2871 | 0.9082 | 0.4363 |

Lowering the classification threshold improves recall but reduces precision. The appropriate threshold depends on the cost of:

- Missing a fraudulent transaction
- Blocking or investigating a legitimate transaction

The notebook selects `0.50` because it provides the best F1-score among the evaluated thresholds.

## Feature Importance

The ten most important features identified by XGBoost were:

| Feature | Importance |
|---|---:|
| V14 | 0.6129 |
| V17 | 0.0530 |
| V12 | 0.0471 |
| V10 | 0.0376 |
| V4 | 0.0321 |
| V7 | 0.0172 |
| V3 | 0.0145 |
| V13 | 0.0128 |
| V1 | 0.0127 |
| V8 | 0.0123 |

`V14` was the most important feature according to the trained model.

Because the `V1`–`V28` variables are anonymized PCA components, their original business meanings are not available.

## Installation

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost jupyter
```

## Usage

1. Download or clone the project.
2. Place `creditcard.csv` in the same folder as the notebook.
3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open and run:

```text
Credit_Card_Fraud_Detection_XGBoost.ipynb
```

If you are using Google Colab, upload `creditcard.csv` when prompted by the notebook.

## Requirements

- Python 3.9 or later
- pandas
- NumPy
- Matplotlib
- scikit-learn
- imbalanced-learn
- XGBoost
- Jupyter Notebook or Google Colab

## Important Notes

- SMOTE is applied only to the training data to avoid test-data leakage.
- Accuracy alone is not an appropriate metric for this dataset because the target classes are highly imbalanced.
- Precision, recall, F1-score, ROC-AUC, and the confusion matrix provide more useful information.
- The test set retains the original class distribution.
- Model performance should be validated on new and recent transaction data before deployment.

## Possible Improvements

Future improvements could include:

- Comparing SMOTE with class-weighted XGBoost
- Tuning XGBoost hyperparameters with cross-validation
- Evaluating precision-recall AUC
- Selecting a threshold using the financial cost of errors
- Calibrating the predicted probabilities
- Using time-based validation instead of a random split
- Investigating data drift
- Adding SHAP explanations
- Comparing XGBoost with LightGBM and CatBoost
- Building a real-time fraud scoring API
- Monitoring false positives and missed fraud after deployment

## Disclaimer

This project is intended for educational and research purposes. It should not be used as the sole system for making financial or fraud-related decisions without further testing, security review, monitoring, and human oversight.

## License

Refer to the original dataset provider for dataset licensing and usage requirements. The project code may be reused according to the license included in the repository.
