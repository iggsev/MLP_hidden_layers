import numpy as np
from activation_function import * 

# Multilayer Perceptron Class  
class MLPerceptron(object):
    
    def __init__(self, sizes,  epochs, lr, phi):
    
        # variable initialization 
        self.epochs = epochs 
        self.n=len(sizes)
        self.lr = lr
        self.sizes= sizes
        self.w=[]
        self.wbias=[]
        self.phi=phi
        
        # randomly generate the weights
        for i in range(0,self.n -1):
            self.wbias.append(np.random.randn(sizes[i+1],1)* np.sqrt(1. / sizes[i+1]))
            self.w.append(np.random.randn(sizes[i+1],sizes[i])* np.sqrt(1. / sizes[i+1]))
        
    def predict(self, x):

        # neural network execution 
        self.foward(x)
        
        #Number of output nodes 
        classification = self.sizes[self.n-1]
        
        # binary classification 
        if classification <= 2:
            return np.where(self.O[self.n-1] >= 0.0, 1, -1)    

        # multiclass classification
        else:
            size=len(x)
            final_answer=np.zeros(size)
            for i in range(0, size):
                larger_index=0
                for j in range(0,classification):
                    if self.O[self.n-1][j,i]>self.O[self.n-1][larger_index,i]:
                        larger_index=j
                    final_answer[i]= larger_index
        
            return final_answer
                     
    # bias node initialization 
    def initbias(self, x): 
        
        N= len(x[0:])
        self.bias= np.ones((1,N))

    # Softmax function
    def softmax(self, x):

        exps = np.exp(x - x.max())
        return exps / np.sum(exps, axis=0)

    # Foward Propagation
    def foward(self, x):
        
        # variable initialization 
        self.initbias(x)
        self.O=[]
        self.Z=[]
        classification = self.sizes[self.n-1]
        self.O.append(x.T)
        
        # go through all layers 
        for i in range(0,self.n -1):
            self.Z.append(np.dot(self.w[i],self.O[i]))
            self.Z[i] += np.dot( self.wbias[i],self.bias)
            # if it's not the last layer 
            if i<self.n -2:
                self.O.append(activation_function(self.Z[i])) 

       # the last layer 

       # binary classification   
        if classification <=2 :    
            self.O.append(activation_function(self.Z[self.n-2]))
        # multiclass classification
        else: 
            self.O.append(self.softmax(self.Z[self.n-2]))

    # Neural Network Training 
    def training(self, x, y):

        # vectors initialization 
        self.loss_vector = [] 
        self.error_vector=[]

        # run through training epochs
        for _ in range(self.epochs):

                # variable initialization 
                loss=[]
                delta=[]
                deltaw=[]
                deltawbias=[]      
                
                # neural network execution  
                self.foward(x[:,])
                loss= y-self.O[self.n-1] #loss calculation 
                
                # save error values
                error= np.mean((abs(self.O[self.n-1]  - y))/2)
                self.error_vector.append(error)
                self.loss_vector.append(np.mean(abs(loss)))

                # BACK PROPAGATION
                
                wbiasnovo=self.wbias
                w_novo= self.w
                for i in range(0,self.n - 1):

                    delta.append( loss* activation_function(self.Z[self.n -2-i],self.phi, derivative=True)) 
                    deltawbias.append(np.dot(self.bias,delta[i].T))
                    wbiasnovo[self.n -2 - i] += self.lr * deltawbias[i].T
                    deltaw.append( np.dot(self.O[self.n -2-i],delta[i].T) )
                    w_novo[self.n -2 - i] += self.lr * deltaw[i].T
                    
                    if (i<self.n-2):
                        loss=np.dot(self.w[self.n-2-i].T, delta[i])

                self.wbias=wbiasnovo    
                self.w=w_novo    
                
        return self