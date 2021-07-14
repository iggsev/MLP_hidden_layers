from scipy import stats
import matplotlib.pyplot as plt
import numpy as np 
from MLPerceptron import *



def histogram(X, y, sizes, epochs, learningrate, phi, N ):
    
    # vector inicialization to save weights
    nhist= len(sizes)
    z=[]
    for hist in range(0,nhist-1):
        z.append([])
        for i in range(0 , sizes[hist]+1):
            z[hist].append([])
     
    # run through all runs
    for i in range(0,N): 
        
        # training neural netwoork 
        ppn = MLPerceptron(sizes, epochs, learningrate, phi)
        ppn.training(X,y)
        
        # save all weights 
        for hist in range(0,nhist-1): 
            a=ppn.w[hist]
            a_bias=ppn.wbias[hist]

            for i in range(0, sizes[hist+1]):

                z[hist][0].append(ppn.wbias[hist][i][0])
                
                for j in range(0, sizes[hist]): 

                    z[hist][j+1].append(a[i][j])

    # histogram configurations
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=N)
    
    # plot histograms for all layers 
    for i in range(0,nhist-1):
        for j in range(0,sizes[i]+1):
            plt.hist(z[i][j],**kwargs, label= "$W_{%d,%d}$" %(i,j) )

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.xlabel('Weight Value ')
        plt.ylabel('Number of Connections')
        plt.show()
    
    

    