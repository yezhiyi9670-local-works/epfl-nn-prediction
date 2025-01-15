import numpy
from abc import abstractmethod, ABCMeta
from typing import Any, Union

class Base(metaclass=ABCMeta):
    def dump_params(self) -> Any:
        '''
        Dump parameters into a serializable object.
        '''
        return None
    
    def load_params(self, data):
        '''
        Load parameters from a formerly dumped object.
        '''
        pass
    
    @abstractmethod
    def forward(self, X: numpy.ndarray) -> numpy.ndarray:
        '''
        Given the input tensor, compute the output.
        '''
        pass
    
    @abstractmethod
    def backward(self, ddY: numpy.ndarray, X: numpy.ndarray, Y: numpy.ndarray) -> numpy.ndarray:
        '''
        Given the original input and the gradient with respect to the output tensor, \\
        compute the gradient with respect to the input
        '''
        pass
    
    def update(self, ddY: numpy.ndarray, X: numpy.ndarray, Y: numpy.ndarray, lr: float):
        '''
        Given the original input and the gradient with respect to the output tensor, \\
        update the params of the current module with Gradient Descent
        '''
        pass

class ReLU(Base):
    def forward(self, X) -> numpy.ndarray:
        return X * (X >= 0)
    def backward(self, ddY, X, Y):
        return ddY * (X >= 0)
    
class LeakyReLU(Base):
    def __init__(self, slope: float):
        self.slope = slope
    def forward(self, X) -> numpy.ndarray:
        return X * (X >= 0) + self.slope * X * (X < 0)
    def backward(self, ddY, X, Y):
        return ddY * (X >= 0) + ddY * self.slope * (X < 0)

class BiasedLinear(Base):
    def __init__(self, in_dims: int, out_dims: int):
        self.weight = numpy.random.normal(0, 0.01, (out_dims, in_dims))
        self.bias = numpy.zeros((out_dims, 1))
    def dump_params(self):
        return [ self.weight, self.bias ]
    def load_params(self, data: Any):
        self.weight, self.bias = data
    def forward(self, X):
        return self.weight @ X + self.bias
    def backward(self, ddY, X, Y):
        return self.weight.transpose() @ ddY
    def update(self, ddY, X, Y, lr):
        self.bias -= numpy.sum(ddY, axis=1, keepdims=True) * lr  # Sum the gradient from all data samples
        ddW = ddY @ X.transpose()
        self.weight -= ddW * lr

class Sigmoid(Base):
    def forward(self, X) -> numpy.ndarray:
        return 1 / (1 + numpy.exp(-X))
    def backward(self, ddY, X, Y):
        return ddY * Y * (1 - Y)

class LossHead(metaclass=ABCMeta):
    @abstractmethod
    def loss(self, estY: numpy.ndarray, gtY: numpy.ndarray) -> float:
        '''
        Compute the loss function given the output and the ground truth.
        '''
        pass
    
    @abstractmethod
    def grad(self, estY: numpy.ndarray, gtY: numpy.ndarray) -> numpy.ndarray:
        '''
        Compute the gradient of the loss with respect to estY.
        '''
        pass

# [TODO] Not working!
# class BCELoss(LossHead):
#     '''
#     Softmax + cross entropy loss head
#     '''
#     def loss(self, estY, gtY):
#         e0 = numpy.exp(estY[0] - estY[0])
#         e1 = numpy.exp(estY[1] - estY[0])
#         return numpy.mean(-numpy.log(
#             (gtY * e1 + (1 - gtY) * e0) / (e0 + e1)
#         ))
#     def grad(self, estY, gtY):
#         e0 = numpy.exp(estY[0] - estY[0])
#         e1 = numpy.exp(estY[1] - estY[0])
#         g0 = e0 / (e0 + e1) - (gtY == 0)
#         g1 = e1 / (e0 + e1) - (gtY == 1)
#         return numpy.vstack([g0, g1]) / gtY.shape[1]

class MSELoss(LossHead):
    def loss(self, estY, gtY):
        return numpy.mean((estY - gtY) ** 2)
    def grad(self, estY, gtY):
        return (estY - gtY) * 2 / gtY.shape[1]

class BinaryFocalLoss(LossHead):
    '''
    Binary focus loss focuses on difficult examples, \\
    thus could potentially deal with unbalanced positive/negative samples.
    
    $
    \\mathrm{L}(p_t) = -\\alpha_t \\dot (1 - p_t)^{\\gamma} \\dot \\log(p_t)
    $
    '''
    
    def __init__(self, positive_weight: float, gamma: float):
        self.positive_weight = positive_weight
        self.gamma = gamma
    
    def loss(self, estY, gtY):
        p_t = 1 - numpy.abs(estY - gtY)
        alpha_t = (
            self.positive_weight * gtY +
            (1 - self.positive_weight) * (1 - gtY)
        )
        return numpy.mean(-alpha_t * (1 - p_t) ** self.gamma * numpy.log(p_t))

    def grad(self, estY, gtY):
        p_t = 1 - numpy.abs(estY - gtY)
        s_t = 2 * gtY - 1  # Sign of estY in p_t
        alpha_t = (
            self.positive_weight * gtY +
            (1 - self.positive_weight) * (1 - gtY)
        )
        return s_t * alpha_t * (
            self.gamma * (1 - p_t) ** (self.gamma - 1) * numpy.log(p_t)
            - (1 - p_t) ** (self.gamma) / p_t
        ) / gtY.shape[1]
