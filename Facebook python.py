# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:48:52 2023

@author: dhair
"""
#*********** Importing Libraries **************

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap

#*********** Import the DataSet *********************

data = pd.read_csv('/Users/dhair/OneDrive/Desktop/Facebook_Ads_2.csv', encoding ='ISO-8859-1')
print(data)
print(data.head(5))
var=6
print(var)

#**************** Visulaizing The dataframe Who are Clicked or not Clicked on Ads **********

click = data[data['Clicked']==1]
no_click = data[data['Clicked']==0]
print('Total number of clicked',len(data))
print('Number of person who Clicked on Ads',len(click))
print('Number of person did not Clicked on Ads',len(no_click))

#**************** Calculate the percentage Click or no click ads ***************

print('% of click',1.*len(click)/len(data)*100)
print('% of Not Click',1.*len(no_click)/len(data)*100)

#********** Now we Draw a ScatterPlot with respect to Time spent and Salary ,on clicked column **************

sns.scatterplot(data['Time Spent on Site'],data['Salary'],hue=data['Clicked'])
plt.show()

#************ Now draw a boxplot for clicked and time spent on ads *************

plt.figure(figsize=(5,5))
sns.boxplot(x='Clicked',y='Salary',data= data)
plt.show()

plt.figure(figsize=(5,5))
sns.boxplot(x='Clicked',y='Time Spent on Site',data= data)
plt.show()

#************ Now draw a Histogram ****************

data['Salary'].hist(bins=40)
plt.show()

data['Time Spent on Site'].hist(bins =20)
plt.show()

#************ Now cleaning The Data set Drop some columns ******************

data.drop(['Names','emails','Country'],axis=1 , inplace = True)
print(data)

#************ Now create x,y column before train test split ****************

X = data.drop('Clicked',axis =1).values
Y = data['Clicked'].values
print(X.shape)
print(Y.shape)

#*************** Now Feature Scaling ***************

sc= StandardScaler()
X = sc.fit_transform(X)
#print(X)

#**************** Model Training ***************

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

print(X_train.shape)

#************ Apply Logistic Regression ****************

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)

#************** Using LR Model Training *****************
Y_predict_train = classifier.predict(X_train)

#************** Apply Confusion Matrix on Training Dataset *****************

cm= confusion_matrix(Y_train, Y_predict_train)
print(cm)
sns.heatmap(cm, annot= True, fmt= 'd')
plt.show()

#***************  Model Testing  *******************

Y_predict_test = classifier.predict(X_test)

#**************** Apply Confusion Matrix on Testing Dataset *******************

cm= confusion_matrix(Y_test, Y_predict_test)
print(cm)
sns.heatmap(cm, annot= True, fmt= 'd')
plt.show()

#************** Now we use k-fold cross Validation  ****************

score = cross_val_score(classifier ,X , Y, cv=10)
print(score)
print(score.mean())


#************ Now creating Classification Report  ****************

print(classification_report(Y_test,Y_predict_test))

#************** Visualising the Training set results  ****************

X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Training set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#************ Visualising the Test  set results **************

X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('magenta', 'blue'))(i), label = j)
plt.title('Facebook Ad: Customer Click Prediction (Testing set)')
plt.xlabel('Time Spent on Site')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()















































