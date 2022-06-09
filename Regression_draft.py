import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


###############################################################################################
#Preprocessing
from sklearn import preprocessing

df = pd.read_csv("xyz.csv")
#df = pd.read_excel("result_V2.xlsx")

x = df.iloc[:,1:2].values
y = df.iloc[:,-1:].values
c = df.iloc[:,-1:].values
a = df.loc[:,"boy":"yas"]
cinsiyet = df.iloc[:,-1].values
ulke = df.iloc[:,0:1].values

#############################################################################################
#LabelEncoder 
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(df.iloc[:,0])

#############################################################################################
#OneHotEncoder 
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

##############################################################################################
#Numpy, datafrme transformation
sonuc1 = pd.DataFrame(data = ulke, index = range(22), columns = ["fr","tr","us"])
sonuc2 = pd.DataFrame(data = a, index = range(22), columns = ["boy","kilo","yas"])
sonuc3 = pd.DataFrame(data = c[:,0:1], index = range(22), columns = ["cinsiyet E=1, K=0"])

#################################################################################################
#DataFrame concat process
s1= pd.concat([sonuc1,sonuc2],axis =1)
s2= pd.concat([s1,sonuc3],axis = 1)


##############################################################################################
#Data Scaling
from sklearn.preprocessing import StandardScaler
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 =StandardScaler()
y_olcekli = sc2.fit_transform(y)

##################################################################################################
#Spliting datas to train and test

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x, y ,test_size=0.33, random_state=0)

##################################################################################################
#Linear Regression

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x ,y)
y_pred = lin_reg.predict(x_test)
print(lin_reg.predict([[11]]))

#################################################################################################
#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
x_poly = poly_reg.fit_transform(x)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

##################################################################################################
#Support Vector Regression

from sklearn.svm import SVR
svr_reg = SVR(kernel ="rbf")
svr_reg.fit(x_olcekli, y_olcekli)
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

##################################################################################################
#Karar Ağacı - Decision Tree

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))

##################################################################################################
#Rondom Forest Regression

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10 , random_state= 0 )
rf_reg.fit(x,y.ravel())
print(rf_reg.predict([[6.6]]))

######################################################################################################
#R2 Squarecalculation
from sklearn.metrics import r2_score

print("R Square: ",r2_score(y, (lin_reg2.predict(poly_reg.fit_transform(x)))))

######################################################################################################
# #P Value Calculation (Probablity Densiy)
import statsmodels.api as sm
model = sm.OLS(lin_reg.predict(x),x)
print(model.fit().summary())

#####################################################################################################
#Correlation Matrix
print(df.corr())

####################################################################################################
#Data Visuolization

plt.scatter(x,y,color ="red")
plt.plot(x,lin_reg2.predict(x_poly), color="blue")
plt.scatter(x_olcekli, y_olcekli, color = "red")
plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color="blue")
plt.scatter(x,y, color="red")
plt.plot(x, r_dt.predict(x), color="blue")

#Saving
import pickle
dosya = "model.kayit"
pickle.dump(lin_reg,open(dosya,"wb"))
yuklenen = pickle.load(open(dosya,"rb"))
print(yuklenen.predict(x_test))





