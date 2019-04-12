from layer import *
import csv
from loss import *
l=MSE()
class Network:
	def __init__(self,batch_size,*args):

	#change List of Arguments into enumeratable object
		EnumOfArguments=enumerate(args,0)
	#Declare Array of instances of 'linear'
		self.Layer=[]
		self.batch_size=batch_size
	#Two variable that will store Dimension and Activation
	#of 'linear' one by one
		Dimension=None
		Activation=None
		Index=0
	#Loop through Enumerator object and Initialise Layer 
	#one by one
		for key,value in EnumOfArguments:
			if(key%2==0):
				Dimension=value
			else:
				Activation=value
				self.Layer.append(linear(Index,batch_size,Dimension[0],Dimension[1],Activation))
				Index+=1
		self.OutputNodes=Dimension[1]


	def forwardpass(self):
			for layer in self.Layer:
				#print(type(layer))
				layer.forward(self.Layer)

	def backwardpass(self,target):
		self.targat=target
		for layer in reversed(self.Layer):
			layer.backward(self.Layer,target)


		for layer in reversed(self.Layer):
			layer.weights -= layer.dl_dw * 0.001
			#print(layer.biases)
			layer.biases-=layer.dl_db*.001
			#print(layer.dl_db)
			#print(layer.biases)
			#print(layer.dl_dw)

	def train(self,file):
		count=0
		self.file_name=file
		with open(self.file_name,'r') as csvfileinput:
			train_data=csv.reader(csvfileinput)
			tag=[[]]
			Target=np.zeros((self.OutputNodes,1))
			#print(Target)
			n=0
			for row in train_data:
				 #print(row)
				 row=list(map(float,row))
				 target=np.zeros((self.OutputNodes,1),dtype='float32')
				 target[int(row[8])][0]=1
				 #print(target)
				 del row[8]
				 row=np.array(row,dtype='float32')
				 row=np.expand_dims(row,axis=0)
				 #print(row)
				 if n==0:
				 		tag=row
				 		Target=target

				 		n+=1
				 elif n<self.batch_size-1:
				 		Target=np.concatenate((Target,target),axis=1)
				 		tag=np.concatenate((tag,row),axis=0)
				 		#print(Target)
				 		n+=1
				 elif n==self.batch_size-1:
				 		Target=np.concatenate((Target,target),axis=1)
				 		#print(row.shape)
				 		#print(tag.shape)
				 		tag=np.concatenate((tag,row),axis=0).transpose()
				 		#print(tag)
				 		#print(Target)
				 		self.Layer[0].s=tag
				 		n=0
				 		#break
				 		#print(tag)
				 		# for i in range(1000):
				 		self.forwardpass()
				 		self.backwardpass(Target)
				 		print(l.compute(self.Layer[len(self.Layer)-1].x,Target))
				 		# print(self.Layer[len(self.Layer)-1].x)
				 		# print(Target)
	def Test(self,file):
		count=0
		self.file_name=file
		with open(self.file_name,'r') as csvfileinput:
			train_data=csv.reader(csvfileinput)
			tag=[[]]
			n=0
			for row in train_data:
				 row=list(map(int,row))
				 Target=np.zeros((self.OutputNodes,1))
				 Target[row[0]][0]=1

				 del row[0]
				 row=np.array(row,dtype='float32')
				 row=np.expand_dims(row,axis=0)
				 if n==0:
				 		tag=row
				 		n+=1
				 elif n<self.batch_size-1:
				 		tag=np.concatenate((tag,row),axis=0)
				 		n+=1
				 elif n==self.batch_size-1:
				 		tag=np.concatenate((tag,row),axis=0).transpose()
				 		self.Layer[0].s=tag
				 	#change the iteration to loop through file
				 		self.forwardpass()
				 		#self.backwardpass(Target)
				 		print(l.compute(self.Layer[len(self.Layer)-1].x,Target))
				 		count+=1
				 			#print(l.compute(self.Layer[len(self.Layer)-1].x,target))
				 		n=0
		print(count)