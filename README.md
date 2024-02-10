### Introduction:
 
- The Bengaluru House Data dataset contains information about various properties in Bengaluru, including details like location, size, number of bathrooms, and price. In this project, we aim to develop a predictive model to estimate the price of properties based on their features. We will preprocess the data, handle missing values and outliers, and then train multiple regression models to predict property prices. Finally, we will evaluate the models' performance and deploy the best-performing one for future predictions
---

```python
#Imported necessary libraries such as NumPy and Pandas.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
```
---
#### 1. Data Transformation:
```python
data=pd.read_csv("Bengaluru_House_Data.csv")
data.head()
```
![alt text](<Screenshot 2024-02-10 180834.png>)
```python
#find null values and fill them with most occuring value
data['location']=data['location'].fillna('Sarjapur Road')
data['size']=data['size'].fillna('2 BHK')
data['bath']=data['bath'].fillna(data['bath'].median())
#split size column and get only first value
#ex. for 2 BHK get only 2
data['size'].str.split()
#store value of no.of bedrooms to bhk column
data['bhk']=data['size'].str.split().str.get(0).astype(int)
```

- Converted range values in the "total_sqft" column to their mean.
```python
#mean
def convertRange(x):
    temp=x.split('-')
    if len(temp)==2:
        return(float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
```

- Apply convertRange() function to total_sqft column.
- Convert price to price per square feet.
- Replace location with count less than 10 with other location.
- Remove those flats whose (data['total_sqft']/data['bhk']) is less than 300ft
- Removed outliers based on square feet per bedroom ("price_per_sqft").
```python
data['total_sqft']=data['total_sqft'].apply(convertRange)

data['price_per_sqft']=data['price']*100000/data['total_sqft']

data['location']=data['location'].apply(lambda x:'other' if x in location_count_less_10 else x)
data['location'].value_counts()

data=data[((data['total_sqft']/data['bhk'])>=300)]
data.describe()
```
---
#### 2. Split Data:
```python
#80% data in X_train and 20%in X_test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
```
---
#### 3. Model Building:

* Pipeline Creation:
Created a data preprocessing pipeline including one-hot encoding and standard scaling.

* Model Training:
Trained Linear Regression, Lasso Regression, and Ridge Regression models.

* Model Evaluation:
Evaluated models' performance using R-squared score.
```python
column_trans = make_column_transformer((OneHotEncoder(), ['location']), remainder='passthrough')

scaler = StandardScaler(with_mean=False)

lr=LinearRegression()

pipe=make_pipeline(column_trans,scaler,lr)

pipe.fit(X_train,y_train)
```

```python
y_pred_lr=pipe.predict(X_test)
r2_score(y_test,y_pred_lr)
```
- Lasso Regression
```python
lasso=Lasso()
pipe=make_pipeline(column_trans,scaler,lasso)
pipe.fit(X_train,y_train)
y_pred_lasso=pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)
```
- Ridge Regression
```python
ridge=Ridge()
pipe=make_pipeline(column_trans,scaler,ridge)
pipe.fit(X_train,y_train)
y_pred_ridge =pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)
```
----
#### 4.Conclusion
```python
print("No Regularization",r2_score(y_test,y_pred_lr))
print("Lasso",r2_score(y_test,y_pred_lasso))
print("Ridge",r2_score(y_test,y_pred_ridge))
```
- r2 score for Ridge regression is higher so we save it using pickle
- This pickle file is used for building a website.