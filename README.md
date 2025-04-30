# AI-Powered Credit Card Fraud Detection and Prevention

## Problem Statement
Credit card fraud leads to massive financial losses every year. This project uses machine learning techniques to detect fraudulent transactions in real time, based on transaction patterns.

## Objectives
- Build classification models to detect fraud.
- Improve recall and F1-score due to class imbalance.
- Provide insights using data visualization.

## Dataset
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Records: 284,807
- Features: 30 (including PCA-transformed ones)

## Tools & Libraries
Python, scikit-learn, pandas, seaborn, matplotlib, XGBoost, imbalanced-learn, Google Colab

## How to Run
```bash
git clone https://github.com/your-username/AI-Credit-Card-Fraud-Detection.git
cd AI-Credit-Card-Fraud-Detection
pip install -r requirements.txt

#Coding
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def scale_features(df):
    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    return df.drop(['Amount', 'Time'], axis=1)