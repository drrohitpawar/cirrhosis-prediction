# Cirrhosis Prediction

Assessing various supervised machine learning models to attempt to accurately predict the presence of cirrhosis based on various provided patient features. 

## Data Overview
- This dataset was downloaded from Kaggle from the following link: https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset.
- The data contains the information collected from the Mayo Clinic trial in primary biliary cirrhosis (PBC) of the liver conducted between 1974 and 1984.
- A total of 424 PBC patients, referred to Mayo Clinic during that ten-year interval, met eligibility criteria for the randomized placebo-controlled trial of the drug D-penicillamine. The first 312 cases in the dataset participated in the randomized trial and contain largely complete data. The additional 112 cases did not participate in the clinical trial but consented to have basic measurements recorded and to be followed for survival. Six of those cases were lost to follow-up shortly after diagnosis, so the data here are on an additional 106 cases as well as the 312 randomized participants.

## Project Overview
-  The aim of the project was to as accurately as possible predict the presence of cirrhosis given patient features.
-  Patient features included: 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Triglycerides', 'Platelets', 'Prothrombin.'
-  Some of these features were continuous numerical variables while others were binary/non-binary categorical variables.
-  The 'Stage' column was the target variable ranging from 1 to 4 with 4 being defined as cirrhosis.
-  As this was a binary classification target variable, ROC AUC score was used to determine the best model.
-  Various classification models were tested and RandomForrestClassifier was found to be the best model.
-  Hyperparameter tuning was performed using GridSearchCV.
-  The best model was found to have a ROC AUC score of 79%

## Project

Packages used: Pandas, NumPy, Scikit-Learn

### Data Cleaning

Null values were examined. Most of the null values came from the additional 106 cases not part of the randomized trail.

```bash
df.isna().sum()
```
Separate lists were created for continuous and categorical features. 

```bash
num_columns = []
for col in df.columns:
  if df[col].dtype == 'int64' or df[col].dtype == 'float64':
    num_columns.append(col)

obj_columns = []
for col in df.columns:
  if df[col].dtype == 'object':
    obj_columns.append(col)
```
SimpleImputer was used to imputer missing values. The numerical values were imputed with the median value of the column. This was used in preference to the mean to ensure that it was skewed by outliers.
For the categorical variables, SimpleImputer was used to impute the most frequent results for the missing values.
```bash
imputer = SimpleImputer(strategy='median', missing_values=np.nan)
imputer.fit(df[num_columns])
df[num_columns] = imputer.transform(df[num_columns])

imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer.fit(df[obj_columns])
df[obj_columns] = imputer.transform(df[obj_columns])
```
