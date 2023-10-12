# credit-risk-classification
Training and evaluating a Logistic Regression model based on loan risk.

## Table of Contents

- [Repository Folders and Contents](#Repository-Folders-and-Contents)
- [Libraries Imported](#Libraries-Imported)
- [Credit Risk Analysis Report](#Credit-Risk-Analysis-Report)
  
## Repository Folders and Contents
- Credit_Risk/Resources/
  - lending_data.csv  --> data csv file
  
- Credit_Risk
  - credit_risk_classification.ipynb  --> Jupyter notebook code

## Libraries Imported
- pandas
- numpy
- pathlib
- confusion_matrix, classification_report from sklearn.metrics
- train_test_split from sklearn.model_selection
- LogisticRegression from sklearn.linear_model

## Credit Risk Analysis Report 
## Overview of the Analysis

In this Challenge we have used `LogisticRegression` to train and evaluate a model based on loan risk. A dataset of historical lending activity from a peer-to-peer lending services company is used to build a model that can identify the creditworthiness of borrowers. </br>
The dataset consists of `loan_size`,	`interest_rate`,	`borrower_income`,	`debt_to_income`,	`num_of_accounts`,	`derogatory_marks`,	 `total_debt` and `loan_status` columns. A value of 0 in the `loan_status` column meant that the loan was healthy and a value of 1 meant that the loan had a high risk of defaulting. Based on this, `loan_status` column was used as label or feature to be predicted, and all the remaining columns from the dataset as known features. </br>

### Splitting the data into training and testing datasets by using train_test_split
Once the two dataframes for labels and features was created, `train_test_split` from `sklearn.model_selection` was used to split the data into training and testing datasets `X_train, X_test, y_train, y_test`
### Fitting a logistic regression model by using the training data
`LogisticRegression` from `sklearn.linear_model` was used to instantiate LogisticRegression Model and fit the training dataset `(X_train, y_train)`. Solver used was `lbfgs`, tried max_iter = 200 and got the same result as default iteration hence decided to allow code to use default iterations. Also random_state of 1 was used in this step.
### Saving the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model
Predictions of testing data labels were then made using testing feature data `X_test` using the fitted model and results compared against actuals: </br> </br>
![image](https://github.com/jyojay/credit-risk-classification/assets/132628129/8d314354-f07c-4c71-9ea4-bed79db3f40b)

### Evaluating the model’s performance by doing the following:
1) Generating a confusion matrix.
2) Printing the classification report.

### Results

**Confusion matrix** </br> </br>
![image](https://github.com/jyojay/credit-risk-classification/assets/132628129/cdc16751-1d77-42f9-acd7-3605cc444dfc)

* Accuracy:  **0.99**
  
* Healthy loan (`loan_status` = 0)
  
    * precision: **1.0**
      
    * recall: **1.0**
      
    * f1-score: **1.0**
      
    * support: **18759**
      
* High-risk loan (`loan_status` = 1)
  
    * precision: **0.87**
      
    * recall: **0.89**
      
    * f1-score: **0.88**
      
    * support: **625**
 
This is summarised in the below **Classification Report** that was generated </br></br>
![image](https://github.com/jyojay/credit-risk-classification/assets/132628129/e0629ba5-ba31-4e2b-a98d-3ed9851a34d9)

### Analysis and Summary

* We have more data with **'0'** (healthy loan) loan-status than  **'1'** (high-risk loan), **18759** and **625** respectively in the dataset hence the accuracy level of **0.99** **is not indicative of the effectiveness of the model**.
* We then look at the **precision**, **recall** and **f1-scores** and the model seems to work perfectly well for **'0'** (healthy loans) as all indicators are **1** or almost **100%** correct.
* However, when we look at the **precision** for **'1'** (high-risk loans), a score of **0.87** shows that **13%** **of the loans that were wrongly predicted as high-risk were actually good loans which might lead to unhappy customers if any action is taken against them.**
* Further when we look at the **recall** which is **0.89 meaning 11% of the loans were wrongly predicted as healthy loans although they were actually high-risk.**
* Hence we can say that this **logistic regression model is not as great at predicting high-risk loans as they are for healthy loans**. **With the understanding that identifying high-risk loans is high priority for the financial organization, this would not be the best model. I would recommend trying methods or combination of methods to handle imbalanced data and compare which performs better for high-risk loan identification in this case.** This is definitely a good begining though.
