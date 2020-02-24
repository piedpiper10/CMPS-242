from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
scaler = MinMaxScaler()
X=pd.read_csv('test_diabetes.csv', delimiter=',', usecols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
normalized_X = preprocessing.normalize(X)
y=pd.read_csv('test_diabetes.csv', delimiter=',', usecols = ['Outcome'])
print len(normalized_X),len(y)
X_train_scaled = scaler.fit_transform(X)
#x=pd.DataFrame(np.concatenate(normalized_X))
df = pd.DataFrame(X_train_scaled)

df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
fin=pd.merge(df,y,left_index=True,right_index=True,how='right')
fin.to_csv('test_diabetes_n.csv',index=False)
