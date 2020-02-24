from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
X=pd.read_csv('train_diabetes.csv', delimiter=',', usecols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y=pd.read_csv('train_diabetes.csv', delimiter=',', usecols = ['Outcome'])
X_test=pd.read_csv('test_diabetes.csv', delimiter=',',usecols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y_test=pd.read_csv('test_diabetes.csv', delimiter=',', usecols = ['Outcome'])

for i in range(1,500,10):
        clf= RandomForestClassifier(n_estimators=i,max_depth=9)
        t0 = time.clock()
        clf = clf.fit(X, y)
        t1=time.clock()-t0
        y_pred_test=clf.predict(X_test)
        y_pred_train=clf.predict(X)
        print "Max Depth:",i
        print "Training Accuracy:",accuracy_score(y, y_pred_train)
        print "Testing Accuracy:",accuracy_score(y_test, y_pred_test)
        print "Running Time:",t1


