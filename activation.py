import numpy as np

class sigmoid():
	"""docstring for sigmoid"""
	def __init__(self):
		pass

	def compute(self,x):
		#print(x)
		return 1/(1+ np.exp(-x))

	def gradient(self,x):
		return self.compute(x) * (1- self.compute(x))
		pass


class ReLU():

 	def __init__(self):
 		pass
 	def compute(self,x):
 		return np.maximum(x,0)

 	def gradient(self,x):
 		return np.where(x <= 0, 0, 1)


class Tanh():
 	def __init__(self):
 		pass

 	def compute(self,x):
 		return np.tanh(x)

 	def gradient(self,x):
 		return 1.0-(np.tanh(x)**2)
#print(a.gradient(np.random.rand(3,1)))
# # x =2
# # print(1+np.exp(-x))
