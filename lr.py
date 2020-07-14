#1 Importing essential libraries
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#2 Importing the dataset

file_name = 'datasets_10624_14867_Salary_Data.csv'
dataset = pd.read_csv(file_name)

#Displaying the dataset
dataset.head(8)

#3 classify dependent and independent variables
X = dataset.iloc[:,:-1].values  #independent variable YearsofExperience
Y = dataset.iloc[:,-1].values  #dependent variable salary

print("\nIdependent Variable (Experience):\n\n", X[:5])
print("\nDependent Variable (Salary):\n\n", Y[:5])

#4 Creating training set and testing set

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.20)


lm = LinearRegression()
lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)

#Plotting Actual observation vs Predictions
plt.scatter(X_test,Y_test, s = 70, label='Actual')
plt.scatter(X_test,Y_pred, s = 90, marker = '^', label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend();
plt.grid();
plt.show();

print(r2_score(Y_test,Y_pred))