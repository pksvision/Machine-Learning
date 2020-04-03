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
		return data[:,0], data[:,1], data[:,2]

	def modify_data(self, x1_data, x2_data, degree):
		array = np.zeros((x1_data.shape[0],2))
		array[:,0] = x1_data
		array[:,1] = x2_data
		return array

	def plot_decision_boundary(self, weight_changes, weights, x_fit_data, x_data, y_data, c_data):
		plt.scatter(x_data, y_data, c=c_data)
		plt.title("Given Data for Logistic Regression")
		plt.xlabel('Marks 1')
		plt.ylabel('Marks 2')		
		
		# new_ys = [np.dot(np.transpose(weights), x_fit_data[i])[0] for i in range(len(x_fit_data))]
		# plt.plot(x_data, new_ys, 'k')
		plt.show()

class LinearRegression(object):
	"""docstring for LinearRegression"""
	def __init__(self, x_data, y_data, degree, iterations, lr):
		super(LinearRegression, self).__init__()
		self.iterations = iterations
		self.alpha = lr
		self.x_data = x_data
		self.y_data = y_data
		self.costs_list = []
		self.degree = degree
		self.activation = Activation()
		self.cost = Cost(len(self.x_data))

	def hypothesis(self, theta, x):
		return self.activation.sigmoid(np.dot(np.transpose(theta), x))
		
	def compute_cost(self, theta, x, y):
		h_x = self.hypothesis(theta, x)
		error = self.cost.error(y, h_x, theta)
		return error, h_x

	def start_regression(self):
		theta = np.random.rand(self.degree, 1)
		print("Init Weights : ", theta)
		weight_changes = np.zeros(( self.iterations+1, self.degree, 1 ))
		weight_changes[0, :, :] =  theta
		for iter in range(self.iterations):
			total_cost=[]
			hypothesis_collection = []
			for item in range(len(self.x_data)):
				cost, hypothesis = self.compute_cost(theta, self.x_data[item], self.y_data[item])
				total_cost.append(cost)
				hypothesis_collection.append(hypothesis)
			# Now have total cost and hypothesis for each item
			etas = [0]*self.degree
			for eta in range(len(etas)):
				for item in range(len(self.x_data)):
					etas[eta] += (hypothesis_collection[item]-self.y_data[item])*(self.x_data[item][eta]) + (self.cost.LAMBDA/self.cost.m)*theta[eta]
			# update weights using etas
			for ind in range(self.degree):
				theta[ind] = theta[ind] - self.alpha* etas[ind]/len(self.x_data)
			weight_changes[iter + 1, :, :] = theta
			print("Iter : ",iter," Avg. error : %.30f "% (sum(total_cost)/len(self.x_data)))
			self.costs_list.append(sum(total_cost)/len(self.x_data))
		return weight_changes, self.costs_list
				
if __name__ == '__main__':
	# degree is 2 
	# Change for polynomial
	degree, iterations, lr = sys.argv[1], sys.argv[2], sys.argv[3]
	data = DataLoader("./data.txt")
	la_data_x, la_data_y, la_data_class = data.load_data()
	data_for_fit_x =  data.modify_data(la_data_x, la_data_y, int(degree))
	la = LinearRegression(data_for_fit_x, la_data_class, int(degree), int(iterations), float(lr))
	weight_changes, costs_list = la.start_regression()
	print("Minimum cost : ", costs_list[costs_list.index(min(costs_list))])
	data.plot_decision_boundary(weight_changes, weight_changes[costs_list.index(min(costs_list))+1], data_for_fit_x, la_data_x, la_data_y, la_data_class)
