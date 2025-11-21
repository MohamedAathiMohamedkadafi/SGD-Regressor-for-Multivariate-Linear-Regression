# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Mohamed Aathil M
RegisterNumber: 25008235
*/
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())


df.info()

X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y=df[['AveOccup','HousingPrice']]
Y.info()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print(Y_pred)


## Output:

<img width="768" height="291" alt="Screenshot 2025-11-21 140043" src="https://github.com/user-attachments/assets/b12773a6-f601-41b8-afed-bcdbddf63514" />




<img width="571" height="330" alt="Screenshot 2025-11-21 140213" src="https://github.com/user-attachments/assets/54d99484-d912-48c0-9087-e66253da3448" />




<img width="459" height="299" alt="Screenshot 2025-11-21 140228" src="https://github.com/user-attachments/assets/e3d4c50f-1f16-49ca-b6ee-0d4beb060b67" />




<img width="272" height="159" alt="Screenshot 2025-11-21 140441" src="https://github.com/user-attachments/assets/ef9c0fe8-3e57-4138-966d-3ece20f9481d" />




<img width="395" height="171" alt="Screenshot 2025-11-21 140450" src="https://github.com/user-attachments/assets/ffb14f32-4944-48cb-b961-c95ecfe8f73f" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
