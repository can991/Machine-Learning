#In this project, the relationship between before and after surface roughness parameters were investigated.
#With this relationship, it is aimed to predict Rz_after from Ra_before to eliminate one measuring parameter.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Preprocessing
from sklearn import preprocessing
veriler = pd.read_excel("roughness.xlsx")
isil = veriler.iloc[:,1:]
Ra_before = isil.iloc[:24,2:3]
# Ra_after = isil.iloc[:23,6:7]
# Rz_before = isil.iloc[:23,5:6]
Rz_after = isil.iloc[:24,5:6]

#Corelation_matrix_creation with matplotlib
corrmatrix = veriler.corr()

#Corelation_matrix_creation with seaborn
sns.heatmap(corrmatrix)

#Spliting datas to train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(Ra_before,Rz_after ,test_size=0.33, random_state=0)


#Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x_train , y_train)
y_pred1 = lin_reg.predict(x_test)

#R2 Square calculation
from sklearn.metrics import r2_score
print("R Square: Linear_Reg",r2_score(y_test, (lin_reg.predict(x_test))))

#Polynomial Regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

for x in range(1,50):
    poly_reg = PolynomialFeatures(degree = x)
    x_poly = poly_reg.fit_transform(Ra_before)
    lin_reg2 = LinearRegression()
    lin_reg2.fit(x_poly,Rz_after)
    if r2_score(y_test,(lin_reg2.predict(poly_reg.fit_transform(x_test)))) > 0.595:
        print("Üs_değeri", x ,"r2_Score", r2_score(y_test,(lin_reg2.predict(poly_reg.fit_transform(x_test)))))

#To predict from Ra_Before to Rz_after         
     
print("If Ra_before = 6.6 , Rz_after prediction = ", lin_reg2.predict(poly_reg.fit_transform([[6.6]])))


#Data Visualization

plt.scatter(Ra_before, Rz_after, color="blue")
plt.title("Blue = Real , Red = Prediction")
plt.xlabel("Ra_before")         
plt.ylabel("Rz_after")
plt.scatter(x_test, lin_reg2.predict(poly_reg.fit_transform(x_test)), color="red")






