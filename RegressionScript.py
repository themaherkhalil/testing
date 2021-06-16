import pandas as pd
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
#from pulp import *
#from lpsolve55 import *

frame = Diabetes_FRAME809082

glucose_levels = frame['Blood_Glucose']
feature_set = frame[['Regular_Insulin_Dose','Basal_Insulin_Dose','Exercise','Meal_Ingestion']]

X_train, X_test, y_train, y_test = train_test_split(feature_set, glucose_levels, test_size=0.4, random_state=0)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
RI = float(reg.coef_[0])
BI = float(reg.coef_[1])
EX = float(reg.coef_[2])
CR = float(reg.coef_[3])
Int = float(reg.intercept_)

print(reg.coef_)
