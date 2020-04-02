# Logistic Regression
# author : Prasen Kumar Sharma
# source : 
import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
import sys

class DataLoader(object):
	"""docstring for DataLoader"""
	def __init__(self, path):
		super(DataLoader, self).__init__()
		self.path = path
		
	def load_data(self):
		path = self.path
		file = open(path)
		lines = file.readlines()
		x_data = []
		y_data = []
		for line in lines:
			xys = line.split(",")
			x_data.append(float(xys[0]))
			y_data.append(float(xys[1].split("\n")[0]))	
		return x_data, y_data

	def plot_data(self, weight_changes, weights, x_fit_data, y_fit_data, x_data, y_data):
		plt.scatter(x_data, y_data)
		plt.title("Given Data for Polynomial Regression")
		plt.xlabel('Population of City in 10,000s')
		plt.ylabel('Profits in $10,000s')		
		new_ys = [sigmoid(np.dot(np.transpose(weights)), x_fit_data[i])[0] for i in range(len(x_fit_data))]
		plt.plot(x_data, new_ys, 'k')
		plt.show()

	def modify_data(self, x_data, degree):
		init_array = []
		for i in range(degree):
			if i is 0:
				init_array.append(np.ones((len(x_data))))
			else:
				init_array.append(np.power(x_data, i))
		x_array = np.transpose(np.array(init_array))
		# x_array = np.transpose(np.array([np.ones((len(x_data))), x_data, np.square(x_data)]))
		return x_array

class LinearRegression(object):
	"""docstring for LinearRegression"""
	def __init__(self, x_data, y_data, degree, iterations, lr):
		super(LinearRegression, self).__init__()
		# model : h(x) = theta_0 + theta_1.x + theta_2.x2 + ... + theta_n-1 xn-1
		self.iterations = iterations
		self.alpha = lr
		self.x_data = x_data
		self.y_data = y_data
		self.costs_list = []
		self.degree = degree

	def compute_cost(self, theta, x, y):
		h_x = sigmoid(np.dot(np.transpose(theta), x))
		root_h_x = h_x - y
		error = root_h_x*root_h_x
		return error

	def start_regression(self):
		theta = np.zeros((self.degree, 1))
		print("Init Weights : ", theta)
		weight_changes = np.zeros(( self.iterations+1, self.degree, 1 ))
		weight_changes[0, :, :] =  theta
		for iter in range(self.iterations):
			total_cost=[]
			for item in range(len(self.x_data)):
				cost = self.compute_cost(theta, self.x_data[item], self.y_data[item])
				total_cost.append(cost)
			etas = [0]*self.degree
			for item in range(len(self.x_data)):
				for eta in range(len(etas)):
					if eta is 0:
						etas[eta] += math.sqrt(total_cost[item])*sigmoid(self.x_data[item][eta])*(1-sigmoid(self.x_data[item][eta]))
					else:
						etas[eta] += math.sqrt(total_cost[item])*sigmoid(self.x_data[item][eta])*(1-sigmoid(self.x_data[item][eta]))*(self.x_data[item][eta])
			for ind in range(self.degree):
				theta[ind] = theta[ind] - self.alpha* etas[ind]/len(self.x_data)
			weight_changes[iter + 1, :, :] = theta
			print("Iter : ",iter," Avg. error : ", sum(total_cost)[0]/len(self.x_data))
			self.costs_list.append(sum(total_cost)[0]/len(self.x_data))
		return weight_changes, self.costs_list
		
if __name__ == '__main__':
	degree, iterations, lr = sys.argv[1], sys.argv[2], sys.argv[3]
	data = DataLoader("./data.txt")
	la_data_x, la_data_y = data.load_data()
	data_for_fit_x =  data.modify_data(la_data_x, int(degree))
	data_for_fit_y = np.array(la_data_y)
	la = LinearRegression(data_for_fit_x, data_for_fit_y, int(degree), int(iterations), float(lr))
	weight_changes, costs_list = la.start_regression()
	# costs_list.index(min(costs_list)) is min index or optimal learned weights
	print("Minimum cost : ", costs_list[costs_list.index(min(costs_list))])
	data.plot_data(weight_changes, weight_changes[costs_list.index(min(costs_list))+1], data_for_fit_x, data_for_fit_y, la_data_x, la_data_y)
