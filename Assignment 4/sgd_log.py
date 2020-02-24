#Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
import pandas as pd
import csv
from matplotlib import pyplot as plt
# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
# Split a dataset into k foldst

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	train =pd.read_csv("train_diabetes_n1.csv")  
	train_set = train.values.tolist()
	test = pd.read_csv("test_diabetes_n1.csv") 
	test_set = test.values.tolist()
	predicted = algorithm(train_set, test_set, *args)
	actual=[]
	for i in range(len(test_set)):
		actual.append(test_set[i][len(test_set[i])-1])
	accuracy = accuracy_metric(actual, predicted)
	return accuracy
 
# Make a prediction with coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
		logl.append(yhat)
	return 1.0 / (1.0 + exp(-yhat))
def predict1(row, coefficients):
        yhat = coefficients[0]
        for i in range(len(row)-1):
                yhat += coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + exp(-yhat))

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
	return coef
 
# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict1(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)
 
# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
filename = 'train_diabetes.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
	str_column_to_float(dataset, i)
# evaluate algorithm
n_folds=1000
logl=[]
for i in range(1,2,5):
	j=i*0.01
	l_rate = j
	n_epoch = 1000
	scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
	print l_rate,":  ",(scores)
plt.plot(logl)
print len(logl)
plt.show()
