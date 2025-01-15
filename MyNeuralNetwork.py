import numpy
import nn_modules
import random

# [TODO] Implement regularization
class MyNeuralNetwork():
    def __init__(self):
        self.modules: list[nn_modules.Base] = [
            nn_modules.BiasedLinear(20, 20),
            nn_modules.ReLU(),
            nn_modules.BiasedLinear(20, 10),
            nn_modules.ReLU(),
            nn_modules.BiasedLinear(10, 1),
            nn_modules.Sigmoid()
        ]
        self.loss_head = nn_modules.BinaryFocalLoss(0.25, 2)
    
    def dump_params(self):
        return [
            module.dump_params()
            for module in self.modules
        ]

    def train(self, X: numpy.ndarray, Y: numpy.ndarray, lr: float, epoch: int):
        '''
        Train the network with the supplied batch.
        
        X shape: (d_in, N); Y shape: (d_out, N)
        
        The supplied batch is fully used in gradient descent.
        '''
        
        trace = ''
        
        # The forward pass
        intermediary: list = [ None ] * (len(self.modules) + 1)  # Gradient Descent requires saving results
        intermediary[0] = X
        # trace += f'=== Forward {0}\n{repr(intermediary[0])}\n'
        for i in range(len(self.modules)):
            intermediary[i + 1] = self.modules[i].forward(intermediary[i])
            # trace += f'=== Module {i}\n{repr(self.modules[i].dump_params())}\n'
            # trace += f'=== Forward {i + 1}\n{repr(intermediary[i + 1])}\n'
            
        # Loss calculation
        loss = self.loss_head.loss(intermediary[-1], Y)
        curr_grad = self.loss_head.grad(intermediary[-1], Y)
        
        # trace += f'=== Grad {len(intermediary) - 1}\n{repr(curr_grad)}\n'

        # Backpropagation
        for i in range(len(self.modules) - 1, -1, -1):
            module = self.modules[i]
            next_grad = module.backward(curr_grad, intermediary[i], intermediary[i + 1])
            module.update(curr_grad, intermediary[i], intermediary[i + 1], lr)
            curr_grad = next_grad
            # trace += f'=== Grad {i}\n{repr(curr_grad)}\n'
        
        # open(f'trace/trace-{epoch}.txt', 'w', encoding='utf-8').write(trace)
        
        return loss
            
    
    def __passthru(self, X: numpy.ndarray):
        for module in self.modules:
            X = module.forward(X)
        return X
    
    def eval(self, X: numpy.ndarray, Y: numpy.ndarray, lr: float):
        estY = self.__passthru(X)
        return self.loss_head.loss(estY, Y)

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
    