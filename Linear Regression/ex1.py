# Linear Regression
# author : Prasen Kumar Sharma
# source : https://github.com/atinesh-s/Coursera-Machine-Learning-Stanford/tree/master/Week%202/Programming%20Assignment/machine-learning-ex1
import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d

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
		plt.title("Given Data for Linear Regression")
		plt.xlabel('Population of City in 10,000s')
		plt.ylabel('Profits in $10,000s')
		plt.plot([x_data[i] for i in range(len(x_data))], 
				[np.dot(np.transpose(weights), x_fit_data[i])[0] for i in range(len(x_data))], '-.k')
		plt.show()
		# Error surface visualization
		weight_0 = []
		weight_1 = []
		for i in range(list(weight_changes.shape)[0]):
			weight_0.append(weight_changes[i,:,0][0])
			weight_1.append(weight_changes[i,:,0][1])
		j_vals = np.zeros(( list(weight_changes.shape)[0], list(weight_changes.shape)[0] ))
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		weight_0, weight_1 = np.meshgrid(np.array(weight_0), np.array(weight_1))
		Z =  np.array([self.compute_error_vis(x,y, x_fit_data,y_fit_data) for x,y in zip(np.ravel(weight_0), np.ravel(weight_1))])
		# print(Z)
		ax.plot_surface(weight_0, weight_1, j_vals, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
		plt.show()

	def compute_error_vis(self, X, Y, x_fit_data, y_fit_data):
		theta = np.zeros((2, 1))
		theta[0,0] = X
		theta[1,0] = Y
		error = 0
		for item in range(len(x_fit_data)):
			h_x = np.dot(np.transpose(theta), x_fit_data[item])
			root_h_x = h_x - y_fit_data[item]
			error += (root_h_x*root_h_x)			
		error /= len(x_fit_data)
		return error

	def modify_data(self, x_data):
		x_array = np.transpose(np.array([np.ones((len(x_data))), x_data]))
		return x_array

class LinearRegression(object):
	"""docstring for LinearRegression"""
	def __init__(self, x_data, y_data):
		super(LinearRegression, self).__init__()
		# model : h(x) = theta_0 + theta_1.x
		self.iterations = 50
		self.alpha = 1e-4
		self.x_data = x_data
		self.y_data = y_data

	def compute_cost(self, theta, x, y):
		h_x = np.dot(np.transpose(theta), x)
		root_h_x = h_x - y
		error = root_h_x*root_h_x
		return error

	def start_regression(self):
		theta = np.ones((2, 1))
		print("Init Weights : ", theta)
		weight_changes = np.zeros(( self.iterations+1, 2, 1 ))
		weight_changes[0, :, :] =  theta
		for iter in range(self.iterations):
			total_cost=[]
			for item in range(len(self.x_data)):
				cost = self.compute_cost(theta, self.x_data[item], self.y_data[item])
				total_cost.append(cost)
			eta_0 = 0
			eta_1 = 0
			for item in range(len(self.x_data)):
				eta_0 += math.sqrt(total_cost[item])
				eta_1 += math.sqrt(total_cost[item])*self.x_data[item][1] 
			theta[0] = theta[0] - self.alpha* eta_0/len(self.x_data)
			theta[1] = theta[1] - self.alpha* eta_1/len(self.x_data)
			weight_changes[iter + 1, :, :] = theta
			print("Iter : ",iter," Avg. error : ", sum(total_cost)[0]/len(self.x_data))
		return weight_changes
		
if __name__ == '__main__':
	data = DataLoader("./ex1data.txt")
	la_data_x, la_data_y = data.load_data()
	data_for_fit_x =  data.modify_data(la_data_x)
	data_for_fit_y = np.array(la_data_y)
	la = LinearRegression(data_for_fit_x, data_for_fit_y)
	weight_changes = la.start_regression()
	data.plot_data(weight_changes, weight_changes[len(weight_changes)-1], data_for_fit_x, data_for_fit_y, la_data_x, la_data_y)
