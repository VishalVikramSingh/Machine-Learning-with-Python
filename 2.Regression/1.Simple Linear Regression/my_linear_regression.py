import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

X_train = X_train.reshape((20,1))
y_train = y_train.reshape((20,1))
X_test = X_test.reshape((10,1))
y_test = y_test.reshape((10,1))


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

##Training Set##
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("salary vs. experience (training set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

##Test Set##
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title("salary vs. experience (test set)")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()