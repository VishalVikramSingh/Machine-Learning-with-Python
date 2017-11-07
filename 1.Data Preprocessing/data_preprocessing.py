                            ## DATA PRE-PROCESSING ##

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('Data.csv')

# matrix of independent variables
X = dataset.iloc[:, 0:3].values       
# vector of dependent variables
y = dataset.iloc[:, 3].values          

                            ## PREPARING THE DATA ##
                            
# handling missing data - replace the missing data with the mean of that column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# imputer = imputer.fit(X[:, 1:3])
# X[:, 1:3] = imputer.transform(X[:,1:3])
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

                            ## ENCODING CATEGORICAL DATA ##
                           
# 'country' and 'purchased' need to be encoded to numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
# 'purchased' column encoding
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

                            ## SPLITTING THE DATASET ##
                            
# splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

                            ## FEATURE SCALING ##

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 		# no need to fit object on test set because already fitted on training set

# since it is a classification problem, no need to do feature scaling on dependent variable y
# np.set_printoptions(threshold = np.nan) to see non-truncated array in ipython
