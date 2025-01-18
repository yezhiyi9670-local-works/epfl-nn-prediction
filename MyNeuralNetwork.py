import numpy
import nn_modules
import random

class MyNeuralNetwork():
    def __init__(self):
        self.modules: list[nn_modules.Base] = [
            nn_modules.BiasedLinear(69, 64, biased=True),
            nn_modules.LeakyReLU(0.01),
            nn_modules.Dropout(p=0.1),
            
            nn_modules.BiasedLinear(64, 32, biased=False),
            nn_modules.BatchNorm(32),
            nn_modules.LeakyReLU(0.01),
            nn_modules.Dropout(p=0.1),
            
            nn_modules.BiasedLinear(32, 32, biased=False),
            nn_modules.BatchNorm(32),
            nn_modules.LeakyReLU(0.01),
            
            nn_modules.BiasedLinear(32, 1, biased=False),
            nn_modules.Sigmoid()
        ]
        self.loss_head = nn_modules.BinaryFocalLoss(0.96, 1)
        self.regularization_factor = 1e-5  # Adjusted: 1e-5 results in much overfitting, while 5e-5 prevents training
    
    def dump_params(self):
        return [
            module.dump_params()
            for module in self.modules
        ]
    
    def load_params(self, data):
        for i in range(len(self.modules)):
            self.modules[i].load_params(data[i])
            
    def loss(self, estY: numpy.ndarray, gtY: numpy.ndarray):
        l2 = 0
        for module in self.modules:
            l2 += module.l2normsq(self.regularization_factor)
        return l2 + self.loss_head.loss(estY, gtY)

    def train(self, X: numpy.ndarray, Y: numpy.ndarray, lr: float, epoch: int):
        '''
        Train the network with the supplied batch.
        
        X shape: (d_in, N); Y shape: (d_out, N)
        
        The supplied batch is fully used in gradient descent.
        '''
        
        # The forward pass
        caches: list = [ None ] * len(self.modules)  # Gradient Descent requires saving results
        curr_value = X
        for i in range(len(self.modules)):
            forward_result = self.modules[i].forward(curr_value, True)
            curr_value = forward_result[0]
            caches[i] = forward_result[1:]
            
        # Loss calculation
        loss = self.loss(curr_value, Y)
        curr_grad = self.loss_head.grad(curr_value, Y)
        
        # Backpropagation
        for i in range(len(self.modules) - 1, -1, -1):
            module = self.modules[i]
            backward_result = module.backward(curr_grad, caches[i])
            curr_grad = backward_result[0]
            module.update(backward_result[1:], lr, self.regularization_factor)
        
        return loss
            
    
    def __passthru(self, X: numpy.ndarray):
        for module in self.modules:
            X = module.forward(X, False)[0]
        return X
    
    def eval(self, X: numpy.ndarray, Y: numpy.ndarray):
        estY = self.__passthru(X)
        return self.loss(estY, Y)

    def predict(self, X: numpy.ndarray):
        '''
        Predict labels for the supplied batch.
        
        Input shape: (d_in, N); Output shape: (d_out, N) 
        '''
        estY = self.__passthru(X)
        return estY

if __name__ == '__main__':
    # Individual test for the neural network
    '''
    The neural network:
        self.modules: list[nn_modules.Base] = [
            nn_modules.BiasedLinear(20, 20),
            nn_modules.ReLU(),
            nn_modules.BiasedLinear(20, 10),
            nn_modules.ReLU(),
            nn_modules.BiasedLinear(10, 1),
            nn_modules.Sigmoid()
        ]
        self.loss_head = nn_modules.BinaryFocalLoss(0.25, 2)
    '''
    
    def gen_data(n):
        X = [ [] for i in range(20) ]
        Y = [ [] ]
        for _ in range(n):
            s = 0
            for i in range(20):
                d = random.randint(0, 1)
                X[i].append(d)
                if i < 5:
                    s ^= d
            Y[0].append(s)
        return numpy.array(X), numpy.array(Y)

    train_X, train_Y = gen_data(1000)
    eval_X, eval_Y = gen_data(1000)
    
    nn = MyNeuralNetwork()
    lr = 0.5
    
    for epoch in range(30001):
        loss = nn.train(train_X, train_Y, lr, epoch)
        pred = nn.predict(train_X)
        win_count = numpy.sum((pred >= 0.5) == train_Y)
        eval_win_count = numpy.sum((nn.predict(eval_X) >= 0.5) == eval_Y)
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss {"%.9f" % loss} train {win_count} eval {eval_win_count}')
        if loss != loss:
            break
        
    open('trace/_pp.txt', 'w', encoding='utf-8').write(repr(nn.dump_params()))
    