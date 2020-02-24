import random
import csv
import pandas as pd
import numpy as np
np.random.seed()
for i in range(100):
	x=random.randint(-100,100)
	y=0.2+2*x+x*x+3*x*x*x
	data=[x,y]
	with open('file100a.csv', mode='a') as file1:
		writer = csv.writer(file1)
    		writer.writerow([x,y])

		
    	

	
