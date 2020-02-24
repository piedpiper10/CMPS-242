from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg=LogisticRegression()

iris=load_iris()
x=iris.data
y=iris.target
logreg.fit(x,y)
a=logreg.predict(x)
print metrics.accuracy_score(y,a)
