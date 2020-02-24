from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
X=pd.read_csv('train_diabetes_n.csv', delimiter=',', usecols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y=pd.read_csv('train_diabetes_n.csv', delimiter=',', usecols = ['Outcome'])
X_test=pd.read_csv('test_diabetes_n.csv', delimiter=',',usecols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y_test=pd.read_csv('test_diabetes_n.csv', delimiter=',', usecols = ['Outcome'])

for i in range(1,200,3):
      		  	clf= MLPClassifier(hidden_layer_sizes=(i,i,i),)
        		t0 = time.clock()
       	 		clf = clf.fit(X, y)
        		t1=time.clock()-t0
        		y_pred_test=clf.predict(X_test)
        		y_pred_train=clf.predict(X)
        		print "Max nodes:",i
        		print "Training Accuracy:",accuracy_score(y, y_pred_train)
       	 		print "Testing Accuracy:",accuracy_score(y_test, y_pred_test)
        		print "Running Time:",t1

