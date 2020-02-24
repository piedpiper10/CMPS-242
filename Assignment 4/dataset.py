import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
X=pd.read_csv('train_diabetes_n.csv', delimiter=',', usecols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y=pd.read_csv('train_diabetes_n.csv', delimiter=',', usecols = ['Outcome'])
print X
print y
X_train, X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=12)
with open('train_diabetes_v3n.csv', 'w') as f:
     pd.concat([X_train, y_train], axis=1).to_csv(f)
with open('train_diabetes_v4n.csv', 'w') as f:
     pd.concat([X_test, y_test], axis=1).to_csv(f)



