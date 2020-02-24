import random
import numpy as np
from scipy.optimize import minimize
np.random.seed()
for i in range(100):
	x= random.sample(xrange(-100,100), 1)
	y=np.random.normal(x,1,100)
	print x,y
