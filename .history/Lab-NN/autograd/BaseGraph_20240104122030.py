from .BaseNode import Node
from typing import List

class Graph(object):
    '''
    计算图类
    '''
    def __init__(self, nodes: List[Node]):
        super().__init__()
        self.nodes = nodes

    def flush(self):        
        for node in self.nodes:
            node.flush()

    def forward(self, X):
        """
        正向传播
        @param X: n*d 输入样本
        @return: 最后一层的输出
        """
        for node in self.nodes:
            X = node.forward(X)
            print(X.shape, X)
        return X

    def backward(self, grad=None):
        """
        反向传播
        @param grad: 前向传播所输出的output对应的梯度
        @return: 无返回值
        """
        # TODO: Please implement the backward function for the computational graph, which can back propagate the gradients
        # from the loss node to the head node.
        for node in reversed(self.nodes):
            grad = node.backward(grad)
            print(grad.shape, grad)
        
        # raise NotImplementedError
    
    def optimstep(self, lr):
        """
        利用计算好的梯度对参数进行更新
        @param lr: 超参数，学习率
        @return: 无返回值
        """  
        # TODO: Please implement the optimstep function for a computational graph, 
        # which can update the parameters of each node based on their gradients.
        for node in self.nodes:
            for i in range(len(node.params)):
                # Update each parameter using gradient descent
                node.params[i] -= lr * node.grad[i]
            # for i, param in enumerate(node.params):
            #     # print(i, param)
            #     if len(node.grad) < i:
            #         raise ValueError(f"Length mismatch in Computation Graph: "
            #                  f"Node gradient has length {len(node.grad)}, but expected {i} .")
            #     if len(node.grad) >= i:
            #         param -= lr * node.grad[i]
        
        # raise NotImplementedError
