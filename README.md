# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")
# Display first few rows
print(df.head())
```
![image](https://github.com/user-attachments/assets/4620d87a-74cd-42b7-99d5-1736ff7ddac6)

```
# Drop missing values
df = df.dropna()
# Find maximum value from Height and Weight feature
print("Max Height:", df["Height"].max())
print("Max Weight:", df["Weight"].max())
```
![image](https://github.com/user-attachments/assets/dc2626ab-b8ca-4720-8cc8-6368f5fd9e8c)
```
# Perform MinMax Scaler
minmax = MinMaxScaler()
df_minmax = minmax.fit_transform(df[["Height", "Weight"]])
print("\nMinMaxScaler Result:\n", df_minmax[:5])
```


![image](https://github.com/user-attachments/assets/ca188301-36d4-49d9-a225-242c8db58f98)

```
# Perform Standard Scaler
standard = StandardScaler()
df_standard = standard.fit_transform(df[["Height", "Weight"]])
print("\nStandardScaler Result:\n", df_standard[:5])
```
![image](https://github.com/user-attachments/assets/456bdc3f-d176-4807-8435-8db4b758d75d)
```
# Perform Normalizer
normalizer = Normalizer()
df_normalized = normalizer.fit_transform(df[["Height", "Weight"]])
print("\nNormalizer Result:\n", df_normalized[:5])
```
![image](https://github.com/user-attachments/assets/675c44f7-6108-43f9-9ea5-95b60d33bd6e)
```
# Perform MaxAbsScaler
max_abs = MaxAbsScaler()
df_maxabs = max_abs.fit_transform(df[["Height", "Weight"]])
print("\nMaxAbsScaler Result:\n", df_maxabs[:5])
```
![image](https://github.com/user-attachments/assets/28b435a7-9458-45ba-a7b9-8a085bb00629)

```
# Perform RobustScaler
robust = RobustScaler()
df_robust = robust.fit_transform(df[["Height", "Weight"]])
print("\nRobustScaler Result:\n", df_robust[:5])
```
![image](https://github.com/user-attachments/assets/4c7125d6-ddbc-481c-be65-96b5c03cc151)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2
# Load Titanic dataset
df = pd.read_csv('/content/titanic_dataset.csv')
# Display column names
print(df.columns)
```
![image](https://github.com/user-attachments/assets/4183fc45-f027-4952-b525-0dcc46c70665)

```
# Show shape of dataset
print("Shape:", df.shape)
```
![image](https://github.com/user-attachments/assets/5567bf5c-771f-4ccc-9b22-8f16af123a0c)
```

# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]
# Drop irrelevant columns
df1 = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
print("Missing Age values before:", df1['Age'].isnull().sum())
```

![image](https://github.com/user-attachments/assets/e4bac951-b2a4-40ea-bf1a-ac9814267441)

```
# Apply SelectKBest for top 3 features
feature = SelectKBest(mutual_info_classif, k=3)
# Reorder columns as required
df1 = df1[['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']]
# Define feature matrix and target vector
X = df1.iloc[:, 0:6]
y = df1.iloc[:, 6]
# Confirm columns
print("X Columns:", X.columns)
y = y.to_frame()
print("y Columns:", y.columns)
```

![image](https://github.com/user-attachments/assets/9ef88b39-6c91-4fde-99bd-962aa1364eb5)
```
# Fit SelectKBest
feature.fit(X, y.values.ravel())

```

![image](https://github.com/user-attachments/assets/7047152c-0214-4068-9765-bee749a1fb9a)

```
# Get selected feature scores
scores = pd.DataFrame({"Feature": X.columns, "Score": feature.scores_})
print("\nFeature Scores:\n", scores.sort_values(by="Score", ascending=False))
```

![image](https://github.com/user-attachments/assets/1c80d0de-34a1-4b28-ba2e-f5d76e7bbd8f)















       # INCLUDE YOUR CODING AND OUTPUT SCREENSHOTS HERE
       
# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
