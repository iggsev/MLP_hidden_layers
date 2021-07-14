import numpy as np

from mlxtend.data import loadlocal_mnist
from mlxtend.data import mnist_data


# NEURAL NETWORK PARAMETERS 
train_sizes=[784, 64,32, 10] # vector representing number of nodes per layer
epochs=1000
learningrate= 0.0005
seed=1


# load database
x,y  = loadlocal_mnist(
        images_path='/Users/PC/Desktop/TCC-final/leitordenumeros/train-images.idx3-ubyte', 
        labels_path='/Users/PC/Desktop/TCC-final/leitordenumeros/train-labels.idx1-ubyte')

# SET PARAMETERS 

# training size
train_size=5000

# ACTIVATION FUNCTION 
# hyperbolic tangent: tanh 
# sigmoid: sig
# sine: sin  

phi="sig"

# graphic options
N_runs= 1 # number of runs for histogram 

import main_number_reader