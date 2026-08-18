# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Define the input and output datasets.
2. Initialize the SGD Regressor model.
3. Train the model using the given data.
4. Predict outputs for the given sample.
5. Display actual and predicted values with graphs.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: VARSHINI K M 
RegisterNumber:  212225240179
*/
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 7],
    [7, 6]
])

Y = np.array([
    [5, 8],
    [6, 9],
    [9, 12],
    [10, 13],
    [13, 16],
    [16, 20],
    [17, 21]
])

sgd = SGDRegressor(
    max_iter=1000,
    eta0=0.01,
    learning_rate='constant',
    random_state=42
)

model = MultiOutputRegressor(sgd)

model.fit(X, Y)

Y_pred = model.predict(X)

print("\nActual Outputs:")
print(Y)

print("\nPredicted Outputs:")
print(np.round(Y_pred, 2))

new_sample = np.array([[8, 7]])
prediction = model.predict(new_sample)

print("\nPrediction for:", new_sample)
print("Predicted House Price:", f"{prediction[0][0]:.2f}")
print("Predicted Number of Occupants:", f"{prediction[0][1]:.2f}")

plt.figure(figsize=(6, 4))
plt.scatter(Y[:, 0], Y_pred[:, 0])
plt.plot(
    [Y[:, 0].min(), Y[:, 0].max()],
    [Y[:, 0].min(), Y[:, 0].max()],
    'r--'
)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("House Price: Actual vs Predicted")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(Y[:, 1], Y_pred[:, 1])
plt.plot(
    [Y[:, 1].min(), Y[:, 1].max()],
    [Y[:, 1].min(), Y[:, 1].max()],
    'r--'
)
plt.xlabel("Actual Number of Occupants")
plt.ylabel("Predicted Number of Occupants")
plt.title("Occupants: Actual vs Predicted")
plt.grid(True)
plt.show()
```

## Output:
<img width="445" height="732" alt="Screenshot 2026-08-18 121653" src="https://github.com/user-attachments/assets/48572098-05bf-4e65-a013-91ec8eabe2b6" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
