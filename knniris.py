 
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
k_range=range(1,26)
scores=[]
for k in k_range:
	knn=KNeighborsClassifier(n_neighbors=k)
	knn.fit(x_train,y_train)
	y_pred=knn.predict(x_test)
	scores.append(metrics.accuracy_score(y_test,y_pred))
plt.plot(k_range, scores)
plt.xlabel("value of k ")
plt.ylabel("testing accuracy")
plt.show()
