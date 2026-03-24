# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.The dataset is loaded, cleaned by removing unnecessary columns, and categorical variables are converted into numerical form using one-hot encoding. 2. The data is split into training and testing sets, and a Linear Regression model is trained to predict car prices. 3. Cross-validation is applied to evaluate model stability, and performance is measured using MSE, MAE, and R² score. 4. A scatter plot of actual vs predicted prices is drawn to visually analyze the accuracy of the model predictions.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
data=pd.read_csv('CarPrice_Assignment.csv')
data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)
x=data.drop('price',axis=1)
y=data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
print("Name: SYED ADIL S")
print("Reg no: 2122252040453")
print("\n=== Cross-Validation ===")
cv_scores=cross_val_score(model,x,y,cv=5)
print("Fold R**2 Scores",[f"{score:.4f}" for score in cv_scores])
print(f"Average R**2:{cv_scores.mean():.4f}")
y_pred=model.predict(x_test)
print("\n=== Test Set Performence ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Prdicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="1099" height="407" alt="Screenshot 2026-03-24 214046" src="https://github.com/user-attachments/assets/38fc4dc0-d424-4af6-a397-4d74b8b48a7e" />

<img width="1060" height="703" alt="Screenshot 2026-03-24 213740" src="https://github.com/user-attachments/assets/d5298378-3056-4ec3-9c9f-e56664214f1d" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
