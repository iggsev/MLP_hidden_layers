import numpy as np 
#from main_barrier import f_barrier

from run_barrier import f_barrier
# Set Definition Function
# Between x=[x_min,x_max] if N points 
def set_def(N,x): 
    
    
    x_min=x[0]
    x_max=x[1]
    X= np.linspace(x_min, x_max, N)
    y_max=(min(f_barrier(X)))-1 
    y_min=(max(f_barrier(X)))+1

    deltax= abs(x_min-x_max)
    deltay= abs(y_min-y_max)

    nx= int( deltax*np.sqrt(N/(deltax*deltay) ) )
    ny= int( deltay*np.sqrt(N/(deltax*deltay) ) )

    N=nx*(ny+1)

    #initialize neural network input variables 
    X=np.zeros((N,2))
    y=np.zeros(N)

    # uniform distribution of points 
    for j in range(0,ny+1):
        yaux2= deltay*j/ny +y_max
        for i in range(0,nx):
            # coordenate x
            X[j*nx+i,0]=deltax*i/nx +x_min 
            #coordenate y
            X[j*nx+i,1]=yaux2
    
    # actual classification of points 
            if f_barrier(X[j*nx+i,0])<= yaux2:
                y[j*nx+i]=1
            else:
                y[j*nx+i]=-1
    
    return X,y 

