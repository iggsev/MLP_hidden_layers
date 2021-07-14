import matplotlib.pyplot as plt
import numpy as np

from MLPerceptron import *
from activation_function import *
from histogram import *

from mlxtend.data import loadlocal_mnist
from mlxtend.data import mnist_data

from run_number_reader import *

############################# DATABASE ADJUST ############################

N=len(x) # data size 

# value normalization 
x = (x/255).astype('float32')
yn=np.zeros((N,10))


# array dimention 

for i in range(0,N):
    yn[i,y[i]]=1

# training set 
x_train=x[0:train_size]
y_train=yn[0:train_size].T

# testing set
x_test=x[train_size:N]
y_test=yn[train_size:N].T


###################### NEURAL NETWORK EXECUTION #########################

np.random.seed(seed) #set seed
# neural network execution

ppn = MLPerceptron(train_sizes, epochs, learningrate, phi)
ppn.training(x_train,y_train)


# training error calculation
final_answer_train = ppn.predict(x_train)
aux=final_answer_train-y[0:train_size]
aux=np.where(aux != 0.0, 1,0)   
train_error=sum(aux)/train_size

# testting error calculation 
ppn.foward(x_test)
final_answer_test = ppn.predict(x_test)
a=final_answer_test-y[train_size:N]
aux=np.where(aux != 0.0, 1,0)   
test_error=sum(aux)/N

######################### PLOTTING RESULTS #############################

# LOSS FUNCTION
plt.yscale(value='log')
plt.plot(range(1, len(ppn.loss_vector) + 1), ppn.loss_vector, marker='o')
plt.title('Loss Function')
plt.xlabel('Epochs')
plt.ylabel(' output - y ')
plt.tight_layout()
plt.show()


# FRACTION ERROR 
plt.plot(range(1, len(ppn.error_vector) + 1), ppn.error_vector, marker='o')
plt.title('Fraction Error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.tight_layout()
plt.show()

# HISTOGRAM OF WEIGHT 

#histogram(x_train, y_train, train_sizes, epochs, learningrate, phi, N_runs )
