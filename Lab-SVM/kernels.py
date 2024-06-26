import numpy as np
np.random.seed(0)

class Base_kernel():
    
    def __init__(self):
        pass
    
    def __call__(self, x1, x2):
        """
        Linear kernel function.
        
        Arguments:
            x1: shape (n1, d)
            x2: shape (n2, d)
            
        Returns:
            y : shape (n1, n2), where y[i, j] = kernel(x1[i], x2[j])
        """
        pass


class Linear_kernel(Base_kernel):
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, x1, x2):
        # TODO: Implement the linear kernel function
        y = np.dot(x1, x2.T)
        return y
    
    
class Polynomial_kernel(Base_kernel):
        
    def __init__(self, degree, c):
        super().__init__()
        self.degree = degree
        self.c = c
        
    def __call__(self, x1, x2):
        # TODO: Implement the polynomial kernel function
        y = (np.dot(x1, x2.T) + self.c) ** self.degree
        return y

class RBF_kernel(Base_kernel):
    
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma 
        
        
    def __call__(self, x1, x2):
        # TODO: Implement the RBF kernel function
        if x1.ndim == 1 and x2.ndim == 1:
            y = np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * self.sigma ** 2))
        elif (x1.ndim > 1) and (x2.ndim > 1):
            y = np.exp(-np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis=2) ** 2 / (2 * self.sigma ** 2))
        else:
            y = np.exp(-np.linalg.norm(x1 - x2, axis=1) ** 2 / (2 * self.sigma ** 2))

        return y