import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
import csv
import pandas as pd
fields = ['X', 'Y']
count=0
def foo():
	global count
	count=count+1
	print count
def lik(parameters):
	'''	
	x5=parameters[0]
	x4=parameters[1]
	'''
	x3=parameters[0]
	x2=parameters[1]
	x1=parameters[2]
	m=parameters[3]
	sigma=parameters[4]
	
	for i in np.arange(0, len(x)):
        	y_exp =x**3*x3+x**2*x2+x*x1+m
		foo()
    	L = (len(x)/2*np.log(2*np.pi)+len(x)/2*np.log(sigma**2)+1/(2*sigma**2)*sum((y-y_exp)**2))
    	#L=sum(np.log(sigma) + 0.5 * np.log(2 * np.pi) + (y - y_exp) ** 2 / (2 * sigma ** 2))
	return L
df = pd.read_csv('file100a.csv', skipinitialspace=True)
print df[['X', 'Y']]
x=df['X']
y=df['Y']
lik_model = minimize(lik, np.array([1,1,1,1,1]))
plt.scatter(x,y)
plt.plot(x, lik_model['x'][0] * x + lik_model['x'][1]+ lik_model['x'][2]+ lik_model['x'][3])
plt.show()
print lik_model['x'][0],lik_model['x'][1],lik_model['x'][2],lik_model['x'][3]

