from .BaseNode import Node
import numpy as np
from .Init import kaiming_normal, kaiming_uniform, zeros

class relu(Node):
    def __init__(self):
        super().__init__("relu")

    def forward(self, x):
        self.cache.append(x)
        return np.clip(x, 0, None)

    def backward(self, grad):
        # x = self.cache.pop()
        x = self.cache[-1]
        return np.multiply(grad, x > 0) 

class Linear(Node):
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias) # Store the parameters in self.params, i.e., weight = self.params[0], bias = self.params[1]


    def forward(self, X):
        '''
        X: input data, dim: (Batch_size, indim), where "Batch_size" is the number of data samples in a batch.
        return: output data, dim: (Batch_size, outdim)
        '''
        # TODO: YOUR CODE HERE
        # if X.shape[1] != self.params[0].shape[0]:
        #     raise ValueError(f"Dimension mismatch in Linear layer: "
        #                      f"Input data has shape {X.shape}, but expected {self.params[0].shape[0]} features based on the weight matrix.")
        # if (X @ self.params[0]).shape[1] != self.params[1].shape[0]:
        #     raise ValueError(f"Dimension mismatch in Linear layer: "
        #                      f"First has shape {(X @ self.params[0]).shape }, but expected {self.params[1].shape}.")
        self.cache.append(X)
        return np.dot(X, self.params[0]) + self.params[1]

        
        # raise NotImplementedError

    def backward(self, grad):
        '''
        grad: gradient of output data, dim: (Batch_size, outdim)
        return: gradient of input data, dim: (Batch_size, indim)
        '''
        # TODO: YOUR CODE HERE, remember to save the gradients of weight and bias in self.grad.
        X = self.cache.pop()
        # X = self.cache[-1]
        # print(X)
        self.grad = [X.T @ grad, np.sum(grad, axis=0)]
        return np.dot(grad,self.params[0].T)

        
        # raise NotImplementedError



class sigmoid(Node):
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, X):
        '''
        X: input data, dim: (*)
        return: output data, dim: (*)
        '''
        # TODO: YOUR CODE HERE
        # X = np.clip(X, -500, 500)
        self.cache.append(X)
        X = np.clip(X, -500, 500)
        return 1 / (1 + np.exp(-X))
        
        # raise NotImplementedError        

    def backward(self, grad):
        '''
        grad: gradient of output data, dim: (*)
        return: gradient of input data, dim: (*)
        '''
        # TODO: YOUR CODE HERE
        # X = self.cache.pop()
        X = self.cache[-1]
        X = np.clip(X, -500, 500)
        sigmoid_output = 1 / (1 + np.exp(-X))
        return grad * sigmoid_output * (1 - sigmoid_output)
        
        # raise NotImplementedError        
        

class MSE(Node):
    def __init__(self):
        super().__init__("MSE")
        
        
    def forward(self, X, Y):
        '''
        X: output of model, dim: (Batch_size, *)
        Y: target data, dim: (Batch_size, *)
        return: MSE loss, dim: (1)
        '''
        # TODO: YOUR CODE HERE
        self.cache.append((X, Y))
        return np.mean((X - Y) ** 2)
        
        # raise NotImplementedError       
    
    
    def backward(self):
        '''
        return: gradient of input data, dim: (Batch_size, *)
        '''
        # TODO: YOUR CODE HERE
        X, Y = self.cache.pop()
        # X, Y = self.cache[-1]
        # n = len(X)
        n = X.shape[0]
        return (2/n) * (X - Y)

        
        # raise NotImplementedError       

