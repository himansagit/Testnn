import numpy as np
import function as F
import activation as a
import loss as l
#batch_size = 2

#batch_size
#activation
##loss
##learning_rate
class linear:
	def __init__(self, index,batch_size, input, output,activation):
		self.batch_size=batch_size
		self.input = input
		self.index = index
		self.output = output
		#self.weights=np.zeros((output,input),dtype='float32')
		self.weights = np.random.rand(output,input) * np.sqrt(2/(input + output)) #they will be initialised using special methods
		self.biases = np.tile((np.random.rand(output,1)), self.batch_size)
		#self.biases=np.tile((np.random.rand(output,1)),batch_size)
		#self.biases=np.tile((np.zeros((output,1),dtype='float32')),batch_size)
		#self.weights = np.random.rand(output,input) * np.sqrt(2/(input + output)) #they will be initialised using special methods
		#self.biases = np.tile((np.random.rand(output,1)), self.batch_size)
		self.s = [[0]]
		self.x = [[0]]
		self.dl_ds = np.ones((output,self.batch_size))
		self.dl_dx = np.ones((output,self.batch_size))
		self.dl_dw = np.ones((output,input))
		self.dl_db = np.ones((output,self.batch_size))
		self.activation = activation
		self.loss = l.MSE()

	def forward(self,net):
		if self.index == 0:
			#print(self.s)
			self.x = self.activation.compute(self.s) # will actually read the given data and apply activation
		else:
			#print((net[self.index-1]).x)
			#print('n')
			#print(self.weights)
			self.s =  self.weights @ (net[self.index-1]).x + self.biases
			self.x =  self.activation.compute(self.s)
		#activation still to be applied
			pass
		pass

	def backward(self,net,expected):#this expected has to be removed and incrporated in net
		if(len(net) == self.index+1):
			# print('gr')
			looo = l.MSE()
			self.dl_dx = looo.gradient(self.x, expected)
			self.dl_ds = self.dl_dx * self.activation.gradient(self.s)
			self.dl_dw = self.dl_ds @ net[self.index-1].x.transpose()
			#self.weights-=self.dl_dw*0.01
			#(net[self.index-1]).dl_dx = self.weights.transpose() @ self.dl_ds
			#print(net[self.index])
			self.dl_db = np.tile(np.matrix(self.dl_ds.mean(1)).transpose(),self.batch_size)

			
			#fatal error in dl_dw, a1b1+ a2b2+ b3B3,, right now ots calculating a1+a2+a3 * b1+b2+b3
			# print(self.dl_dx)
			# print("^")
		else:
			# print("Hello")
			self.dl_dx=(net[self.index+1].weights.transpose())@(net[self.index+1].dl_ds)
			self.dl_ds = self.dl_dx * self.activation.gradient(self.s)
			#if(self.index != 0):
				#(net[self.index-1]).dl_dx = self.weights.transpose() @ self.dl_ds

			self.dl_db = np.tile(np.matrix(self.dl_ds.mean(1)).transpose(),self.batch_size)

			#self.dl_db = self.dl_ds
			if(self.index != 0):
				self.dl_dw = self.dl_ds @ net[self.index-1].x.transpose()


# net= []

# # expected = np.random.rand(4,1)
# x = linear(0,1,3)
# #print(x.activation.compute(np.random.rand(3,1)))
# net.append(x)
# y = linear(1,3,10)
# z = linear(2,10,2)
# net.append(y)
# net.append(z)
# siii = a.sigmoid()


# #print(z.biases)
# #print(x.x)
# #print(expected)
# for i in range(100):
# 	#x.forward(net)
# 	# Input (temp, rainfall, humidity)
# 	x.x= inputs = np.array([[1, 1, 1],[1,0,0]], dtype='float32').transpose()

# # # Targets (apples, oranges)
# 	expected = np.array([[0.23, 0.73], [0.98, 0.00]], dtype='float32').transpose()


# 	y.forward(net)
# 	z.forward(net)

# 	z.backward(net,expected)
# 	y.backward(net,expected)
# 	z.weights -= z.dl_dw 
# 	y.weights -= y.dl_dw  
# 	z.biases -= z.dl_db 
# 	y.biases -= y.dl_db 
	
# 	# hi = l.MSE()
# 	# print(hi.compute(z.x,expected))
	
 	
# 	#print(you.compute(z.x, expected))


# # x.forward(net)

# # Input (temp, rainfall, humidity)
# x.x= inputs = np.array([[1, 0, 0],[1,1,1]], dtype='float32').transpose()
# # # Targets (apples, oranges)
# #expected = np.array([[0.23, 0.68]], dtype='float32').transpose()


# y.forward(net)
# z.forward(net)
#print(z.x)
# print(expected)
# #print(you.gradient(y.x, expected))
# y.backward(net,expected)
# print(y.dl_dx)
#print(y.x)
# print(expected)
# qwe = a.sigmoid()
# print(qwe.compute(expected))
# y.printA()
# print(y.dl_dx)
# print(y.printA())
# print("\n")
# los = l.MSE()
# print(los.gradient(y.x,expected))
# print(y.x)
# a = np.random.rand(3,1)
# y = np.random.rand(1,1)
#print(len(net)) ##this is how you find length of list in python
