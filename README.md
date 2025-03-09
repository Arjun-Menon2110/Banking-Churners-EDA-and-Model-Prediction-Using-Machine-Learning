# Bank Churners Prediction Using Machine Learning

## Overview
This project is an end-to-end machine learning workflow for predicting customer churn in a bank. It includes **Exploratory Data Analysis (EDA), feature engineering, model training, evaluation, and predictions** using Logistic Regression and K-Nearest Neighbors (KNN).

---

## Dataset Description
The dataset contains **10,127** records with **21** columns:
- **Customer Demographics**: `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category`
- **Banking Relationship Features**: `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`
- **Financial Attributes**: `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`
- **Transaction Behavior**: `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio`
- **Target Variable**: `Attrition_Flag` (0: Existing Customer, 1: Attrited Customer)

---

## Exploratory Data Analysis (EDA)
### 1. Data Cleaning
- No missing values were found.
- Removed duplicates.
- Verified data types and value distributions.

### 2. Statistical Insights
- **Customer churn rate**: **16.07%**
- **Average age of churned customers**: **46.66 years**
- **Gender distribution among churned customers**: **57% Female, 43% Male**
- **Top education levels for churned customers**: Graduate (487), High School (306), Uneducated (237)
- **Dependent count for churned vs. existing customers**:
  - Churned: **2.40**
  - Existing: **2.34**
- **Transaction behavior**:
  - Churned customers had **lower transaction counts and amounts**.
  - They had **fewer bank relationships** and higher inactivity.

### 3. Correlation Analysis
- **Features most correlated with churn**:
  - `Contacts_Count_12_mon` (**0.20**)
  - `Months_Inactive_12_mon` (**0.15**)
  - `Total_Ct_Chng_Q4_Q1` (**-0.25**)
  - `Total_Trans_Ct` (**-0.35**)

---

## Feature Engineering
### 1. Encoding Categorical Variables
- `Attrition_Flag`: Converted to binary (1 = Attrited, 0 = Existing)
- `Gender`: 1 (Male), 0 (Female)
- One-hot encoding applied to:
  - `Education_Level`, `Marital_Status`, `Card_Category`

### 2. Feature Transformation
- Income levels mapped to ordinal values:
  - `Unknown`: 0, `Less than $40K`: 1, `$40K - $60K`: 2, `$60K - $80K`: 3, `$80K - $120K`: 4, `$120K+`: 5
- Log transformation applied to skewed features:
  - `Credit_Limit`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Ct_Chng_Q4_Q1`

### 3. Scaling and Outlier Removal
- `RobustScaler` used to normalize numerical features.
- Outliers removed using **Interquartile Range (IQR)**.

### 4. Final Selected Features
- `Total_Relationship_Count`, `Total_Revolving_Bal`, `Total_Ct_Chng_Q4_Q1`
- `Total_Trans_Ct`, `Contacts_Count_12_mon`, `Months_Inactive_12_mon`

---

## Model Training and Evaluation

### **1. Logistic Regression**
- **Oversampling**: SMOTE used for class balance.
- **Accuracy**: **88.25%**
- **Confusion Matrix:**
  ```
  [[1563  136]
   [ 102  225]]
  ```
- **Metrics:**
  - Precision: **94% (Class 0), 62% (Class 1)**
  - Recall: **92% (Class 0), 69% (Class 1)**
  - F1-score: **93% (Class 0), 65% (Class 1)**

### **2. K-Nearest Neighbors (KNN)**
- **Hyperparameter tuning**: Best `k = 1`
- **Accuracy**: **88.25%**
- **Confusion Matrix:**
  ```
  [[1585  114]
   [ 124  203]]
  ```
- **Metrics:**
  - Precision: **93% (Class 0), 64% (Class 1)**
  - Recall: **93% (Class 0), 62% (Class 1)**
  - F1-score: **93% (Class 0), 63% (Class 1)**

---

## Predictions
### Example 1: Customer likely to stay
```
Total_Relationship_Count: 5
Total_Revolving_Bal: 777
Total_Ct_Chng_Q4_Q1: 1.625
Total_Trans_Ct: 42
Contacts_Count_12_mon: 3
Months_Inactive_12_mon: 1
Prediction: Will Stay
```

### Example 2: Customer likely to churn
```
Total_Relationship_Count: 2
Total_Revolving_Bal: 0
Total_Ct_Chng_Q4_Q1: 0.6
Total_Trans_Ct: 16
Contacts_Count_12_mon: 3
Months_Inactive_12_mon: 3
Prediction: Will Attrite
```

---

## Conclusion
- **Both Logistic Regression and KNN achieved 88.25% accuracy.**
- **KNN performed slightly better in recall for churn detection.**
- **Future improvements**:
  - Hyperparameter tuning for Logistic Regression.
  - Experiment with ensemble models (Random Forest, Gradient Boosting).
  - Feature engineering to extract more meaningful patterns.

---

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy pandas scikit-learn seaborn matplotlib imbalanced-learn
```

---

## How to Run
1. Load the dataset (`BankChurners.csv`).
2. Run the **EDA script** to explore data and visualize trends.
3. Execute the **preprocessing script** to clean and transform data.
4. Train the models using Logistic Regression and KNN.
5. Evaluate model performance using accuracy, confusion matrix, and classification report.
6. Predict customer churn based on new data.

---

## Author
- **Arjun**

