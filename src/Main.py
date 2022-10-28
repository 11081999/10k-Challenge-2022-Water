#Step 0: Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#Step 1: Data Preprocessing

#!  Notice how this time every value is numeric so there is no need for Encoding categorical data.
#!  Nor is there a need to fill the missing values.

dataset = pd.read_excel(r'C:\Users\Roberto\Desktop\10k-Challenge-2022-Water\src\Datasets\TemperaturasNL.xlsx')
# X: data
X = dataset.iloc[ : , 0].values
# Y: Target (single dim array)
Y = dataset.iloc[ : , 2].values

print(X)
print(Y)
print("Len", len(X), len(Y))

#Step 2: Fitting playnomial Regression Model to the set
mymodel = np.poly1d(np.polyfit(X, Y, 10)) 
myline = np.linspace(1, 34, 100) 

#Step 3: Visualization
plt.scatter(X, Y, color= 'pink')
plt.plot(myline, mymodel(myline))
plt.show()

#Step 4: Calculate R^2

#?  R^2 tells us how much of the variation in "Y_AXYS"-
#?  can be explained by "X_AXYS"

print("R^2", r2_score(Y, mymodel(X))) 
