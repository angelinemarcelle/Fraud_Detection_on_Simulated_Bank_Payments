# Fraud Detection with Machine Learning
This repository contains code for fraud detection using machine learning algorithms. The code demonstrates various steps involved in the fraud detection process, including data preprocessing, exploratory data analysis, feature engineering, model training, and evaluation.

###Dataset
The dataset used for fraud detection is named "banksim_data.csv". It contains information about various transactions, including customer details, merchant details, transaction amounts, categories, and a fraud label indicating whether the transaction is fraudulent or not.

The dataset can be accessed from the following URL: [banksim_data.csv](https://shorturl.at/yAIMW) - Google Drive

### Code Overview
The code performs the following steps:

1. Imports necessary libraries for data analysis and machine learning.
2. Loads the dataset from the provided URL using pandas.
3. Preprocesses the data by dropping unnecessary columns and converting categorical features into dummy variables.
4. Explores the dataset by analyzing unique values, visualizing data distributions, and computing statistics.
5. Balances the dataset using SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
6. Splits the dataset into training and testing sets.
7. Trains and evaluates different machine learning models, including Logistic Regression, K-Nearest Neighbors, and Random Forest.
8. Calculates the cross-validation scores and displays the mean accuracy.
9. Generates ROC-AUC curves to visualize the model's performance.
10. Calculates the base score, which represents the accuracy achieved by predicting all transactions as non-fraudulent.

### Usage
To use this code, follow these steps:

1. Install the required libraries by running `pip install -r requirements.txt`.
2. Execute the code step-by-step or run the entire script.

## Conclusion
The code provides an insight of fraud detection using various machine learning techniques and data visualization tools. 
Please note that there are a lot of imbalance in the data due to the very small fraudulent payment count compared to non-fraudulent. Hence, the numbers may seem very high which is not always the case with different types of datasets.
