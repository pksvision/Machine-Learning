# Logistic Regression
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
import sys

class Activation(object):
	"""docstring for Activation"""
	def __init__(self):
		super(Activation, self).__init__()
		self.EPSILON = 1e-5
		self.maximum = 1- self.EPSILON
		self.minimum = self.EPSILON

	def clamp(self, x):
		return max(self.minimum, min(x, self.maximum))

	def sigmoid(self, x):
		return self.clamp(1/(1+np.exp(-x)))

	def sigmoid_derivative(self, x):
		return self.sigmoid(x) * (1- self.sigmoid(x))

class Cost(object):
	"""docstring for Cost"""
	def __init__(self, m):
		super(Cost, self).__init__()
		self.LAMBDA = 1
		self.m = m

	def error(self, y_true, hypothesis, theta):
		regularized = 0
		for j in range(len(theta)):
			regularized += math.pow(theta[j],2)
		return -((y_true)*math.log(hypothesis) + (1-y_true)*math.log(1-hypothesis)) + regularized*self.LAMBDA/(2*self.m) 
	
class DataLoader(object):
	"""docstring for DataLoader"""
	def __init__(self, path):
		super(DataLoader, self).__init__()
		self.path = path
		
	def load_data(self):
		data = np.loadtxt(self.path, delimiter=',')
		return data

	def modify_data(self, data):
		# first n-1 are features and last channel is class labels
		data_for_fit = np.ones_like(data)
		data_for_fit[:,1:] = data[:, :data.shape[1]-1]
		class_labels = data[:, data.shape[1]-1]
		return data_for_fit, class_labels

	def plot_decision_boundary(self, final_weights, x_fit_data, data):
		plt.scatter(data[:,0], data[:,1], c=data[:,2])
		plt.title("Given Data for Logistic Regression")
		plt.xlabel('Marks 1')
		plt.ylabel('Marks 2')		
		final_weights.reshape(3)
		plot_x = np.array([ min(data[:,0]), max(data[:,1]) ])
		plot_y = (-1/final_weights[2]) *(final_weights[1]*plot_x + final_weights[0])
		plt.plot(plot_x, plot_y, label='Decision Boundary')
		plt.show()

class LinearRegression(object):
	"""docstring for LinearRegression"""
	def __init__(self, x_data, y_data, iterations, lr):
		super(LinearRegression, self).__init__()
		self.iterations = iterations
		self.alpha = lr
		self.x_data = x_data
		self.y_data = y_data
		self.costs_list = []
		self.activation = Activation()
		self.cost = Cost(len(self.x_data))

	def hypothesis(self, theta, x):
		return self.activation.sigmoid(np.dot(np.transpose(theta), x))
		
	def compute_cost(self, theta, x, y):
		h_x = self.hypothesis(theta, x)
		error = self.cost.error(y, h_x, theta)
		return error, h_x

	def start_regression(self):
		theta = np.random.rand(self.x_data.shape[1], 1)
		print("Init Weights : ", theta)
		weight_changes = np.zeros(( self.iterations+1, self.x_data.shape[1], 1 ))
		weight_changes[0, :, :] =  theta
		for iter in range(self.iterations):
			total_cost=[]
			hypothesis_collection = []
			for item in range(len(self.x_data)):
				cost, hypothesis = self.compute_cost(theta, self.x_data[item], self.y_data[item])
				total_cost.append(cost)
				hypothesis_collection.append(hypothesis)
			etas = [0]*self.x_data.shape[1]
			for eta in range(len(etas)):
				for item in range(len(self.x_data)):
					etas[eta] += (hypothesis_collection[item]-self.y_data[item])*(self.x_data[item][eta]) + (self.cost.LAMBDA/self.cost.m)*theta[eta]
			for ind in range(self.x_data.shape[1]):
				theta[ind] = theta[ind] - self.alpha* etas[ind]/len(self.x_data)
			weight_changes[iter + 1, :, :] = theta
			print("Iter : ",iter," Avg. error : %.30f "% (sum(total_cost)/len(self.x_data)))
			self.costs_list.append(sum(total_cost)/len(self.x_data))
		return weight_changes, self.costs_list
				
if __name__ == '__main__':
	# Change for polynomial
	iterations, lr = sys.argv[1], sys.argv[2]
	data = DataLoader("./data.txt")
	la_data = data.load_data()	
	data_for_fit, class_labels =  data.modify_data(la_data)
	la = LinearRegression(data_for_fit, class_labels, int(iterations), float(lr))
	weight_changes, costs_list = la.start_regression()
	print("Minimum cost : ", costs_list[costs_list.index(min(costs_list))])
	data.plot_decision_boundary(weight_changes[costs_list.index(min(costs_list))+1], data_for_fit, la_data)
