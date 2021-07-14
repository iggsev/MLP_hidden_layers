import numpy as np


def activation_function(x,phi="tanh", derivative=False ): 
    if phi=="tanh": 
        return tanh(x, derivative)
    if phi=="sin":
        return sin(x, derivative)
    if phi=="sig":
        return sig(x,derivative)

def tanh( x, derivative=False):
    t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    if derivative:
        aux= np.exp(x)+np.exp(-x)
        dt= 4/(aux)**2
        return dt
    return t

def sin(x, derivative=False):
        t= np.sin(x)
        if derivative:
            dt=np.cos(x)
            return dt
        return t
    
def sig(x, derivative=False):
    t= 1/(1 + np.exp(-x))
    if derivative:
        return  t*(1-t)
    return t