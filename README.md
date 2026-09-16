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
| Fraud |
