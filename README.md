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
Columns that were irrelevant or would cause data leakage were deleted from the dataframe.
```bash
df.drop(['Status', 'N_Days', 'ID'], axis=1, inplace=True)
```

### Pre-processing
The categorical variables were turned into a numerical format for machine learning pre-processing by using the pandas get_dummies function.
The drop first parameter was set to true to avoid unnecessary columns.
```bash
df_dummy = pd.get_dummies(df, drop_first=True)
```
The stage variable was converted to either 0 or 1 to present the presence or non-presence of cirrhosis.
```bash
df_dummy['Stage'] = np.where(df_dummy['Stage'] == 4.0, 1, 0)
```
The data was then ready for modeling and was split into separate features and target variables and split into train and test datasets with 20% test size.
```bash
X = df_dummy.drop('Stage', axis=1)
y= df_dummy['Stage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
```

### Modeling
Initially, KNeighborsClassifier, GradientBoostingClassifier, and RandomForrestClassifier were tested. RandomForrestClassifier performed the best with the highest AUC score.
```bash
knn = KNeighborsClassifier()
gbc = GradientBoostingClassifier()
rf = RandomForestClassifier()
models = [('KNeighborsClassifier', knn), ('GradientBoostingClassifier', gbc), ('RandomForestClassifier', rf)]

for name, model in models:
  model.fit(X_train, y_train)
  y_pred = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, y_pred)
  print(name + ' - AUC score is: ' + str(auc_score))
```
Then hyperparameter tuning was performed. First, a parameter grid was created with 4 parameters
```bash
param_grid = {
  'max_depth' : list(range(1,11)),
  'n_estimators' : [50,100,150,200],
  'min_samples_leaf' : list(range(1,6)),
  'min_samples_split' : list(range(2,6))
}
```
GridSearchCV was used for cross-validation with our parameters in the parameter grid.
```bash
gridCV = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring='roc_auc')
gridCV.fit(X_train, y_train)
y_pred = gridCV.predict_proba(X_test)
```
RandomForrestClassifier was the classification model to use and with hypertuned parameters provided and AUC score of 79%.

Best Parameters:
- max_depth=3,
- min_samples_split=3

## Limitations
I think the model is limited due to a limited dataset. Only 424 samples were available with some of the feature columns having 100+ missing values.
