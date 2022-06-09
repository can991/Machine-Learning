import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #3D görselleştirme
from sklearn import datasets #2D görselleştirme

veriler = pd.read_csv("veriler.csv")
##################################################################################################
#Preprocessing
x = veriler.iloc[: , 1:4].values
y = veriler.iloc[:,4:].values

##################################################################################################
#Spliting datas to train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x, y,test_size=0.33, random_state=0)



###################################################################################################
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state= 0 )
logr.fit(x_train,y_train)
y_pred = logr.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Logistic_Regression")
print(cm)

#####################################################################################################
# KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier( n_neighbors=1, metric = "minkowski")
knn.fit(x_train , y_train)
y_pred = knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Kneighbor")
print(cm)

#####################################################################################################
#Support_Vector_Classifier

from sklearn.svm import SVC
svc = SVC(kernel ="poly")
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("SCV")
print(cm)

######################################################################################################
#Naive_Bayes_Classifier

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Naive_Bayes")
print(cm)

########################################################################################################
#Decision_Tree_Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion ="entropy")
dtc.fit(x_train,y_train)
y_pred = dtc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Decision Tree")
print(cm)

########################################################################################################
#Random_Forest_Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Random Forest")
print(cm)

########################################################################################################
#XGB_Classifier

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("XGBoost Classifier")
print(cm)

######################################################################################################
#ROC , TPR, FPR calculation

y_proba = rfc.predict_proba(x_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
print(thold)


#####################################################################################################
#Cross Validation

from sklearn.model_selection import cross_val_score

basari = cross_val_score(estimator = svc , X= x_train, y=y_train , cv = 4 )
print(basari.mean())
print(basari.std())


######################################################################################################
#Parameter optimization and algorithm selection
from sklearn.model_selection import GridSearchCV

p = [{"kernel": ["linear","rbf","polly"]},
     {"kernel": ["rbf"], "gamma":[1,0.5,0.1,0.01]}]
gs = GridSearchCV(estimator = svc ,
                  param_grid = p,
                  scoring = "accuracy",
                  cv = 10,
                  n_jobs = -1)

grid_search = gs.fit(x_train, y_train)
eniyisonuc = grid_search.best_score_
eniyiparametre = grid_search.best_params_

print(eniyisonuc)
print(eniyiparametre)