import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from activation_function import *


def plot_decision_regions(X, y, classifier,  resolution=0.01):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    x1 =  xx1.ravel()
    x2 = xx2.ravel()
    my={'x': x1, 'y': x2}
    df1 = pd.DataFrame(my)
    aux = df1.iloc[0:len(x1), [0, 1]].values

    classifier.predict(aux)
    Z  = classifier.O[classifier.n-1]
    Z = Z.reshape(xx1.shape)
  
    plt.subplot(121)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap,antialiased=True )
    X[:,0]= sorted(X[:,0])
    
    


def plot_deformation(X, y, classifier,  resolution=0.01): 
    
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, xxx= np.meshgrid(np.arange(x1_min, x1_max, 0.005),
                            np.arange(x2_min, x2_max, 0.005))

    x1 =  xx.ravel()
    x2 = xxx.ravel()
    my={'x': x1, 'y': x2}
    df1 = pd.DataFrame(my)
    aux = df1.iloc[0:len(x1), [0, 1]].values
    
    for i in range(0 , classifier.sizes[1]-1): 
        deform_x = activation_function(np.dot(classifier.w[0], aux.T)+classifier.wbias[0])
        def_x1 = deform_x[i].reshape(xx.shape)
        def_x2 = deform_x[i+1].reshape(xx.shape)

        Z  = classifier.predict(aux)
        Z= Z.reshape(xx.shape)
    
        plt.subplot(122)
        plt.contourf(def_x1, def_x2, Z, alpha=0.4, cmap=cmap)

        plt.xlabel('Node $%d$ ' %i )
        plt.ylabel('Node $%d$ ' %(i+1)  )
        plt.show() 