from activation import *
from network import *
import csv

Target=np.array([[0.23, 0.73]], dtype='float32').transpose()
Sig=sigmoid()
R=ReLU()

row=np.array([[1, 1]], dtype='float32').transpose()


#Network class -- batch size followed by dimension and activation fucntion for each layer
NN=Network(2, (1,8),Sig, (8,10), Sig, (10,2),Sig)

NN.train('diabetesdata.csv')

# NN=Network(1,(1,2),Sig,(2,10),Sig,(10,4),Sig)
# inputs=np.random.rand(2,1)
# target=np.random.rand(4,1)
# NN.Layer[0].s=inputs
# NN.forwardpass()
# NN.backwardpass(target)
#NN.Test('mnist_test.csv')
			 
#print(tag)
# #print(row.shape)
# Target=np.array([[0.23, 0.73], [0.98, 0.00]], dtype='float32').transpose()

# NN=Network(row,Target,(1,3),Sig,(3,10),Sig,(10,2),Sig)
# NN.train()
# NN.Test()

#In final version data object will pass to train() function