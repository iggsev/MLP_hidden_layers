from activation_function import * 


def precision(X ,y , w, wbias):
    from main_barrier import f_barrier
    N= len(X[0:])
    bias= np.ones((1,N))
    n=len(w)
    
    erros=[]
    for j in range(0,10):
        
            O=[]
            Z=[]
            O.append(X.T)
        
            for i in range(0,n):
            
                Z.append(np.dot(w[i],O[i]))
                Z[i] += np.dot( wbias[i],bias)
                O.append(activation_function(Z[i]))

            output=O[n]
            error= y- output          
            erros.append(np.mean(abs(error/2)))
            X[:,1]=(X[:,1]+f_barrier(X[:,0]))/2

    return erros

