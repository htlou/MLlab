import numpy as np
from typing import Tuple, List
# from vis_util import plot_training_progress

import matplotlib.pyplot as plt



def plot_progress(i: int, loss_history: List[float], w_module_history: List[float], x: np.ndarray, y: np.ndarray, w: np.ndarray):
    # clear current axes
    plt.clf()
    
    # draw loss history
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, label='Loss')
    plt.title('Loss History')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('Loss')
    plt.legend()

    # draw w history
    plt.subplot(1, 3, 2)
    plt.plot(w_module_history, label='w module')
    plt.title('w model History')
    plt.xlabel('Iterations (x10)')
    plt.ylabel('w module')
    plt.legend()

    if x.shape[1] == 3:  # 2 features + 1 bias
        plt.subplot(1, 3, 3)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 50)
        yy = np.linspace(ylim[0], ylim[1], 50)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = np.dot(np.c_[xy, np.ones(xy.shape[0])], w)
        Z = Z.reshape(XX.shape)

        
        # draw decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
        plt.title('Decision Boundary')
    

    plt.tight_layout()
    plt.pause(0.01)  # pause a bit so that plots are updated
    

def sigmoid(x):
    ''' 
    Sigmoid function.
    '''
    return 1 / (1 + np.exp(-x))

class LogisticRegression():
    ''' 
    Logistic Regression
    '''
    def __init__(
        self,
    ) -> None:
        
        self.w = None # random intialize w
        self.lr = None # learning rate
        self.reg = None # regularization parameter

    def predict(
        self,
        x: np.array
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        ''' 
        Logistic Regression (LR) prediction.
        
        Arguments:
            x : (n, d + 1), where n represents the number of samples, d the number of features

        Return:
            prob: (n,), LR probabilities, where prob[i] is the probability P(y=1|x,w) for x[i], from [0, 1]
            pred: (n,), LR predictions, where pred[i] is the prediction for x[i], from {-1, 1}

        '''
        # implement predict method,
        # !! Asumme that : self.w is already given.

        # TODO: first, you should add bias term to x
        # print(x.shape)
        x_bias = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

        
        # TODO: second, you should compute the probability by invoking sigmoid function
        z = np.dot(x_bias, self.w)
        prob = sigmoid(z)

        # TODO: third, you should compute the prediction (W^T * x >= 0 --> y = 1, else y = -1)
        pred = np.where(prob >= 0.5, 1, -1)
        # print("prob & Pred")
        # print(prob, pred)
        

        return prob, pred

        
    
    

    def fit(
        self,
        x: np.array,
        y: np.array,
        n_iter: int,
        lr: float,
        reg: float,
    ) -> None:
        ''' 
        Logistic Regression (LR) training.

        Arguments:
            x : (n, d + 1), where n represents the number of training samples, d the number of features
            y : (n,), where n represents the number of samples
            n_iter : number of iteration
            lr : learning rate
            reg : regularization parameter
            
        Return:
            None
        '''
        self.lr = lr
        self.reg = reg
        
        x = np.concatenate((x,np.ones((x.shape[0], 1))), axis=1) # add bias term
        self.w = np.random.normal(0, 1, x.shape[1]) # random intialize w
        loss_history = []
        w_module_history = []

        
        for i in range(n_iter):
            
            # TODO: firstly, compute the loss with regularization term
            loss = self.calLossReg(x, y)

            # TODO: secondly, update the weight 
            self.update(x, y)

            # plot loss and w module every 10 iterations
            if i % 10 == 0:
                
                loss_history.append(loss)

                w_module_history.append(np.linalg.norm(self.w))
                print("iter: {}, loss: {}, w_module: {}".format(i, loss, w_module_history[-1]))
                plot_progress(i, loss_history,w_module_history, x, y, self.w)
    
        

    def update(
        self,
        x: np.array,
        y: np.array,
    ) -> None:
        
        '''
        Update the parameters--weight w
        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            None
        '''

        # implement gradient descent algorithm

    
        # TODO: 1. compute the gradient
        m = x.shape[0]
        z = np.dot(x, self.w)
        # gradient = np.dot(x.T, (sigmoid(z) - y)) / m + self.reg * self.w
        p_hat = sigmoid(z)

        # Transform y from {-1, 1} to {0, 1}
        transformed_y = (y + 1) / 2

        # Compute gradient
        gradient = np.dot(x.T, (p_hat - transformed_y)) / m 
        gradient[:-1] += (self.reg / m) * self.w[:-1]
        # print(gradient)

        # TODO: 2. update the weight 
        self.w -= self.lr * gradient



    def calLossReg(
        self,
        x: np.array,
        y: np.array,
    ):
        ''' 
        Compute the loss

        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            loss: float, the loss value
        '''
        # TODO: compute the Logistic Regression loss, including regularization term
        # !! Note that the label y is from {-1, 1}
        m = x.shape[0]
        z = np.dot(x, self.w)
        # loss = -np.mean(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z)))
        # reg_loss = loss + (self.reg / (2 * m)) * np.sum(self.w**2)
        p_hat = sigmoid(z)
        transformed_y = (y + 1) / 2  # Transforming y from {-1, 1} to {0, 1}
        loss = -np.mean(transformed_y * np.log(p_hat) +
                         (1 - transformed_y) * np.log(1 - p_hat))

        # L2 Regularization (excluding the bias term)
        reg_loss = loss + (self.reg / (2 * m)) * np.sum(self.w[:-1]**2)

        return reg_loss



    def calLoss(
        self,
        x: np.array,
        y: np.array,
    ):
        ''' 
        Compute the loss

        Arguments:
            x: (n, d+1), training samples, where n represents the number of training samples, d the number of features
            y: (n,), training labels, where n represents the number of training samples

        Return:
            loss: float, the loss value
        '''
        # TODO: compute the Logistic Regression loss, including regularization term
        # !! Note that the label y is from {-1, 1}
        m = x.shape[0]
        z = np.dot(x, self.w)
        p_hat = sigmoid(z)
        transformed_y = (y + 1) / 2  # Transforming y from {-1, 1} to {0, 1}
        loss = -np.mean(transformed_y * np.log(p_hat) + 
                        (1 - transformed_y) * np.log(1 - p_hat))
        reg_loss = loss + (self.reg / (2 * m)) * np.sum(self.w[:-1]**2)

        return reg_loss


from data_generator import gen_2D_dataset

x_train, y_train = gen_2D_dataset(100, 100, noise = 0)
x_test, y_test = gen_2D_dataset(50, 50, noise = 0.7)


LR = LogisticRegression()
LR.fit(x_train, y_train, lr=0.1, n_iter=300,reg=0.1)