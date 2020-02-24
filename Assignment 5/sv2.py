import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
np.random.seed(0)
X = np.r_[np.random.randn(4, 2) - [2, 2], np.random.randn(4, 2) + [2, 2]]
print type(X)
a=[[1,1],[1,2],[2,1],[0,1],[1,0],[0,0]]
X=np.array(a) 

print X
Y = [1] * 3 + [-1] * 3

fig, ax = plt.subplots()
clf2 = svm.LinearSVC(C=1).fit(X, Y)
#clf2=SVC();
#clf2.fit(X,Y)
#print clf2.support_vectors_ 


# get the separating hyperplane
w = clf2.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf2.intercept_[0]) / w[1]
print clf2
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),
                     np.arange(y_min, y_max, .2))
Z = clf2.predict(np.c_[xx2.ravel(), yy2.ravel()])

Z = Z.reshape(xx2.shape)
ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=25)
ax.plot(xx,yy)


ax.axis([x_min, x_max,y_min, y_max])
plt.show()
