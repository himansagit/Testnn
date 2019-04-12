import numpy as np

class MSE:
	"""docstring for MSE"""
	def __init__(self):# x is prediction , y is desired output
		pass
	

	def compute(self,x,y):
		error = (x-y)*(x-y)
		error=np.mean(error)
		return error
		pass

	def gradient(self,x,y):
		return (2*(x-y))
		#return (x-y)
	





# #for testing purpose

# x = np.random.rand(3,1)
# y = np.random.rand(3,1)

# a = MSE(x
# # t= np.random.rand(1,1)
# print(x)
# print(y)
# z = a.compute()
# # print((x-y)*(x-y))

# print(z)
# # print(z@t)

# g =a.gradient()
# print(g)
