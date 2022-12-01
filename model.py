# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 12:38:17 2022

@author: FLEET
"""
#importing Libraries
import numpy as np   
np.random.seed(42)   ## so that output would be same
import pandas as pd
import seaborn as sns
#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#Evaluation
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve
#for warning
from warnings import filterwarnings
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
filterwarnings("ignore")  ## To remove any kind of warning

#----------------------------------------------------------------------------------------------------------#
# Data Import and cleaning

#importing the dataset

df = pd.read_csv('donors.csv')

# Check for missing values

df.isnull().values.any()

#------------------------------------------------------------------------------------------------------#
# Setting variables

data2 = df.drop(df.columns[[0]],axis = 1)

data = data2.copy()                          # Create copy of DataFrame
data = data2.fillna(data2.mean())            # Mean imputation

data.isnull().values.any()

#scale the data for easier analysis

scaler = MinMaxScaler()
data = scaler.fit_transform(data)


#we can then define the last attribute as our target variable 

target = pd.DataFrame(df['Category'])

#convert the data and target into NumPy arrays

data =  np.array(data, dtype= float)
target =  np.array(target, dtype= float)

#split original DataFrame into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2, random_state=42)

#-------------------------------------------------------------------------------------------------------------#

## Build an model (Logistic Regression)
log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train,y_train);
## Evaluating the model
log_reg = log_reg.score(x_test,y_test)
## Build an model (KNN)
knn = KNeighborsClassifier()
knn.fit(x_train,y_train);
## Evaluating the model
knn = knn.score(x_test,y_test)
## Build an model (Random forest classifier)
clf= RandomForestClassifier()
clf.fit(x_train,y_train);
## Evaluating the model
clf = clf.score(x_test,y_test)
## Build an model (Support Vector Machine)
svm = SVC()
svm.fit(x_train,y_train)
## Evaluating the model
svm = svm.score(x_test,y_test)


#-------------------------------------------------------------------------------------------------------------#
log_reg_grid = {'C': np.logspace(-4,4,30),
"solver":["liblinear"]}
#setup  the gird cv
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                           verbose=True)
#fit grid search cv
gs_log_reg.fit(x_train,y_train)
score = gs_log_reg.score(x_test,y_test)



#------------------------------------------------------------------------------------------------------------#
y_preds = gs_log_reg.predict(x_test)
y_preds

#-----------------------------------------------------------------------------------------------------------#
import pickle
# Save trained model to file
pickle.dump(gs_log_reg, open("ensamble_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))


















