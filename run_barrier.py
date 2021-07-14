import numpy as np 


# NEURAL NETWORK PARAMETERS 
sizes= [2,2, 1] # vector representing number of nodes per layer
epochs=1000
learningrate= 0.01
seed=1


# Barrier Function
def f_barrier(x):
    return abs(x)

# SET PARAMETERS 

#Training
N=1000 # number of points 
x= [-1,1] # domain


#Testing
N_test=1000 # number of points 
x_test= [-5,5] # domain 

# ACTIVATION FUNCTION 
# hyperbolic tangent: tanh 
# sigmoid: sig
# sine: sin  

phi="tanh"

# graphic options

resolution= 0.01
N_runs= 100 # number of runs for histogram 

import main_barrier 
