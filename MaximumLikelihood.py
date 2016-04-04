from __future__ import division
from numpy import *
import pylab as plt
from scipy.optimize import minimize
import math


class Minimiser:

	def __init__(self, datafile):
		# loading data with a file
		self.data = loadtxt(datafile)


	# probability density function
	def pdf(self, var, t):	
		# returning the pdf function for given variables
		return var[0]*(1.0/var[1])*exp(-t/ var[1]) + (1 - var[0])*(1.0/var[2])*exp(-t/var[2])


	# determines the maximum likelihood - negative log likelihood
	# by summing all the logs of the pdf for the values in datafile
	def max_likelihood(self, var):
		return  -1 * sum ( [log(self.pdf(var, t)) for t in self.data] )

			
	# uses the scipy package optimize.minimize to find minimum values of f, tau1, tau2
	# returns the minimum values
	def min(self):	
		return minimize(self.max_likelihood, [0.8, 0.2, 1.3], method = 'L-BFGS-B', bounds = [(0, 1), (0.001, 10),  (0.001, 10)] ).x	


	# method for determining the errors on f, tau1 and tau2
	def error(self, increment):
		# calling min method to determine the minimun values in the pdf
		min_value = self.min()
		# calling method to determine maximum likelihood at the min values
		opt = self.max_likelihood(min_value)

		f_min, f_max, tau1_min, tau1_max, tau2_min, tau2_max = 0, 0, 0, 0, 0, 0


		for i in range(1000):

			# condition for upper errors in f
			# if the maximum likelihood, when incresasing the function from its min f value in increment sizes, 
			# reaches a value of 0.5 then set upper f_max, this is never updated again as f_max is no longer zero
			if self.max_likelihood([min_value[0] + i*increment, min_value[1], min_value[2]]) - opt >= 0.5 and f_max == 0:
				f_max = i * increment

			# same method used for the lower error in f, 
			# except now increasing the the function in increment sizes in the other direction 
			if self.max_likelihood([min_value[0] - i*increment, min_value[1], min_value[2]]) - opt >= 0.5 and f_min == 0:
				f_min = i * increment

			# upper error for tau1
			if self.max_likelihood([ min_value[0], min_value[1] + i*increment, min_value[2]]) - opt >= 0.5 and tau1_max == 0:
				tau1_max =  i * increment

			# lower error for tau1
			if self.max_likelihood([ min_value[0], min_value[1] - i*increment, min_value[2]]) - opt >= 0.5 and tau1_min == 0:
				tau1_min =  i * increment

			# upper error for tau2
			if self.max_likelihood([ min_value[0], min_value[1], min_value[2] + i*increment]) - opt >= 0.5 and tau2_max == 0:
				tau2_max =  i * increment

			# lower error for tau2
			if self.max_likelihood([ min_value[0], min_value[1], min_value[2] - i*increment]) - opt >= 0.5 and tau2_min == 0:
				tau2_min =  i *increment

			# exists the loop when the quanities are no longer zero
			if tau1_max !=0  and tau2_max !=0  and tau1_min !=0  and tau2_min !=0  and f_min !=0  and f_max != 0:
				break


		# returning a dictonary 
		return {'f':    [f_min, min_value[0] , f_max],
				'tau1': [tau1_min, min_value[1], tau1_max],
				'tau2': [tau2_min, min_value[2], tau2_max]}








		