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
    def forward(self, X: numpy.ndarray, is_train: bool) -> list[numpy.ndarray]:
        '''
        Given the input tensor, return a list of tensors.
        
        The first one is the output, while the rest are cache tensors that can be used in the backward pass.
        '''
        pass
    
    @abstractmethod
    def backward(self, ddY: numpy.ndarray, cache: list[numpy.ndarray]) -> list[numpy.ndarray]:
        '''
        Given the gradient with respect to the output tensor and the cache created before \\
        return a list of tensors
        
        The first one is the gradient with respect to the input, while the rest are values used to update the params.
        '''
        pass
    
    def update(self, updaters: list[numpy.ndarray], lr: float, reg_factor: float):
        '''
        Given the updater created in the backward pass, \\
        update the params of the current module with Gradient Descent
        '''
        pass
    
    def l2normsq(self, reg_factor: float) -> float:
        '''
        Calculate L-2 norm of the params. For regularization.
        '''
        return 0

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

# ================ Activation functions ================

class ReLU(Base):
    def forward(self, X, is_train):
        return [ X * (X >= 0), X ]
    def backward(self, ddY, cache):
        [ X ] = cache
        return [ ddY * (X >= 0) ]
    
class LeakyReLU(Base):
    def __init__(self, slope: float):
        self.slope = slope
    def forward(self, X, is_train):
        return [ X * (X >= 0) + self.slope * X * (X < 0), X ]
    def backward(self, ddY, cache):
        [ X ] = cache
        return [ ddY * (X >= 0) + ddY * self.slope * (X < 0) ]

# [TODO] Make this numerically safe
class Sigmoid(Base):
    def forward(self, X, is_train):
        Y = 1 / (1 + numpy.exp(-X))
        return [ Y, Y ]
    def backward(self, ddY, cache):
        [ Y ] = cache
        return [ ddY * Y * (1 - Y) ]

# ================ Parameterized layers ================

class BiasedLinear(Base):
    def __init__(self, in_dims: int, out_dims: int, biased: bool = True):
        self.weight = numpy.random.normal(0, 1 / in_dims, (out_dims, in_dims))
        self.bias = numpy.zeros((out_dims, 1))
        self.biased = biased
    def dump_params(self):
        return [ self.weight, self.bias ]
    def load_params(self, data: Any):
        self.weight, self.bias = data
    
    def forward(self, X, is_train):
        Y = self.weight @ X + self.bias
        return [ Y, X ]
    def backward(self, ddY, cache):
        [ X ] = cache
        ddB = numpy.sum(ddY, axis=1, keepdims=True)
        ddW = ddY @ X.transpose()
        return [ self.weight.transpose() @ ddY, ddB, ddW ]

    def update(self, updaters, lr, reg_factor):
        [ ddB, ddW ] = updaters
        self.bias -= self.bias * lr * 2 * reg_factor + ddB * lr
        self.weight -= self.weight * lr * 2 * reg_factor + ddW * lr
    def l2normsq(self, reg_factor = 0):
        return reg_factor * (numpy.sum(self.bias ** 2) + numpy.sum(self.weight ** 2))

# ================ Optimization layers ================

#[TODO] Fixed implementation
class BatchNorm(Base):
    '''
    The Batch Normalization layer, see https://zhuanlan.zhihu.com/p/45614576
    '''
    def __init__(self, dims):
        # Output scaling params
        # self.weight = numpy.random.normal(0, 1, (dims, 1))
        self.weight = numpy.ones((dims, 1))
        self.bias = numpy.zeros((dims, 1))
        # Moving average for inference stage
        self.moving_avg = numpy.zeros((dims, 1))
        self.moving_devi = numpy.ones((dims, 1))
        # Constants
        self.MOMENTUM = 0.9
        self.EPS = 1e-5
    def dump_params(self):
        return [ self.weight, self.bias, self.moving_avg, self.moving_devi ]
    def load_params(self, data: Any):
        [ self.weight, self.bias, self.moving_avg, self.moving_devi ] = data
    
    def forward(self, X, is_train):
        if is_train:
            avg_X = numpy.mean(X, axis=1, keepdims=True)
            sqdevi_X = numpy.var(X, axis=1, keepdims=True)
            self.moving_avg = self.moving_avg * self.MOMENTUM + avg_X * (1 - self.MOMENTUM)
            self.moving_devi = self.moving_devi * self.MOMENTUM + sqdevi_X * (1 - self.MOMENTUM)
        else:
            avg_X = self.moving_avg
            sqdevi_X = self.moving_devi
        mult = (self.EPS + sqdevi_X) ** 0.5
        X_normed = (X - avg_X) / mult
        return [ self.weight * X_normed + self.bias, mult, X_normed ]
    
    def backward(self, ddY, cache):
        [ mult, X_normed ] = cache
        
        ddW = numpy.sum(ddY * X_normed, axis=1, keepdims=True)
        ddB = numpy.sum(ddY, axis=1, keepdims=True)
        ddX_normed = self.weight * ddY
        
        N = ddY.shape[1]
        ddX = (
            N * ddX_normed
            - numpy.sum(ddX_normed, axis=1, keepdims=True)
            - X_normed * numpy.sum(ddX_normed * X_normed, axis=1, keepdims=True)
        )
        ddX *= 1/N / mult
        
        return [ ddX, ddW, ddB ]
        
    def update(self, updaters, lr: float, reg_factor: float = 0):
        [ ddW, ddB ] = updaters
        self.bias -= self.bias * lr * 2 * reg_factor + ddB * lr
        self.weight -= self.weight * lr * 2 * reg_factor + ddW * lr
        
    def l2normsq(self, reg_factor):
        return reg_factor * (numpy.sum(self.weight ** 2) + numpy.sum(self.bias ** 2))

class Dropout(Base):
    def __init__(self, p: float = 0.5):
        assert 0 <= p < 1, "Dropout probability must be between 0 and 1"
        self.p = p
        self.mask = None
    
    def forward(self, X: numpy.ndarray, is_train: bool) -> list[numpy.ndarray]:
        if is_train:
            self.mask = (numpy.random.rand(*X.shape) >= self.p)
            self.mask = self.mask / (1 - self.p)
            return [X * self.mask, self.mask]
        else:
            return [X, numpy.ones_like(X)]
    
    def backward(self, ddY: numpy.ndarray, cache: list[numpy.ndarray]) -> list[numpy.ndarray]:
        [mask] = cache
        return [ddY * mask]

# ================ Loss functions ================

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

class CrossEntropyLoss(LossHead):
    """
    Optimized Binary Cross Entropy Loss with improved numerical stability
    """
    def __init__(self, pos_weight: float = 1.0):
        self.pos_weight = pos_weight
        self.eps = 1e-12
        
    def loss(self, estY: numpy.ndarray, gtY: numpy.ndarray) -> float:
        # Clip values for numerical stability
        estY = numpy.clip(estY, self.eps, 1.0 - self.eps)
        # Compute weighted BCE loss
        bce_loss = -self.pos_weight * gtY * numpy.log(estY) - \
                    (1 - self.pos_weight) * (1 - gtY) * numpy.log(1 - estY)
        return numpy.mean(bce_loss)
    
    def grad(self, estY: numpy.ndarray, gtY: numpy.ndarray) -> numpy.ndarray:
        # Clip predictions for numerical stability
        estY = numpy.clip(estY, self.eps, 1.0 - self.eps)
        # Compute gradient
        grad = (-self.pos_weight * gtY / estY + \
                 (1 - self.pos_weight) * (1 - gtY) / (1 - estY)) / gtY.shape[1]
        return grad

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
