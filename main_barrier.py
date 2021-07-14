import matplotlib.pyplot as plt
import numpy as np
import scipy as sp 
import math 


from MLPerceptron import *
from pdr import *
from set import *
from activation_function import *
from error import precision
from histogram import *
from run import * 


###################### NEURAL NETWORK EXECUTION #########################

np.random.seed(seed) #set seed 

# create dataset  
X,y=set_def(N,x) 
X_test, y_test= set_def(N_test, x_test)

# neural network execution 
ppn = MLPerceptron(sizes, epochs, learningrate, phi)
ppn.training(X,y)


######################### PLOTTING RESULTS #############################

# TRAINING PLOT   

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


plot_decision_regions(X, y, classifier=ppn, resolution=resolution)
plt.plot(X[:,0], f_barrier(X[:,0]), 'g--')
plt.show()
plot_deformation(X, y, classifier=ppn)
plt.tight_layout()
plt.show()


erros= precision(X, y, ppn.w, ppn.wbias)
plt.title('Training classification resolution')
plt.xlabel('Resolution')
plt.ylabel('Error')
plt.plot(["$2^0$","$2^1$","$2^2$","$2^3$","$2^4$","$2^5$","$2^6$","$2^7$","$2^8$","$2^9$"], erros, marker='o')      
plt.show()


# TEST PLOT      

plot_decision_regions(X_test, y_test, classifier=ppn, resolution=resolution )
plt.plot(X_test[:,0], f_barrier(X_test[:,0]), 'g--')
plt.tight_layout()
plt.show()


erros=  precision(X_test, y_test, ppn.w, ppn.wbias)
plt.title('Test classification resolution ')
plt.xlabel('Resolution')
plt.ylabel('Error')
plt.plot(["$2^0$","$2^1$","$2^2$","$2^3$","$2^4$","$2^5$","$2^6$","$2^7$","$2^8$","$2^9$"], erros, marker='o')      
plt.show()

# HISTOGRAM OF WEIGHT 

histogram(X, y, sizes, epochs, learningrate, phi, N_runs )
