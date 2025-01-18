from MyNeuralNetwork import MyNeuralNetwork

import time
import os
import numpy
import nn_modules
import random
import pickle
from formula import f1_score

class ReduceLROnPlateau:
    def __init__(self, base_lr, factor=0.1, patience=5, min_lr=1e-5):
        self.base_lr = base_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = numpy.inf
        self.wait = 0

    def step(self, curr_loss):
        if curr_loss < self.best_loss:
            self.best_loss = curr_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.base_lr = max(self.base_lr * self.factor, self.min_lr)
                self.wait = 0
        return self.base_lr

class TrainingController():
    def __init__(self, nn: MyNeuralNetwork):
        self.nn = nn
        self.lr_scheduler = ReduceLROnPlateau(base_lr=32, factor=0.5, patience=4, min_lr=5e-3)
    def train(self, X_train, Y_train, X_eval, Y_eval):
        epoch = -1
        batch_size = 5000
        
        if Y_eval is not None:
            eval_positive_rate = numpy.sum(Y_eval) / Y_eval.size
            print(f'INFO: Evaluation set positive rate is {"%.4f" % eval_positive_rate}')
        
        termination_flag_msg = 'Make change to this file and save to terminate training.'
        os.makedirs('trace/', exist_ok=True)
        open('trace/termination_flag', 'w', encoding='utf-8').write(termination_flag_msg + '\n')
        print('Ready to start. In theory, the training will stop automatically at some moment.')
        print('Edit the contents in trace/termination_flag enforce an early stop.')
        time.sleep(2)
        
        while True:
            epoch += 1
            should_log = (epoch % 1 == 0)
            
            collective_loss = 0
            lr = self.lr_scheduler.base_lr
            
            if epoch > 0:
                if termination_flag_msg != open('trace/termination_flag', 'r', encoding='utf-8', errors='ignore').read().strip():
                    print('Termination flag change detected. Terminating...')
                    break
                
                train_data = numpy.vstack([ X_train, Y_train ]).transpose()
                numpy.random.shuffle(train_data)
                n_batches = train_data.shape[0] // batch_size + 1  # Batch size ~
                batches: list[numpy.ndarray] = numpy.array_split(train_data, n_batches, axis=0)
                
                for batch in batches:
                    if batch.shape[0] == 0:
                        continue
                    batch = batch.transpose()
                    batch_X = batch[:69]
                    batch_Y = batch[69:70]
                    collective_loss += batch_X.shape[1] * self.nn.train(batch_X, batch_Y, lr, epoch)
                    
                collective_loss /= train_data.shape[0]
                
            if collective_loss != collective_loss:
                print('[ ! ] The training loss is NaN. Numerical collapse has occured.')
                print('      The training will stop now.')
                print('      The saved model will be CORRUPT. Do NOT use it for evaluation. Re-train a new one instead.')
                break
            
            eval_loss = 0
            eval_f1 = 0
            eval_correct_rate = 0
            if X_eval is not None:
                eval_loss = self.nn.eval(X_eval, Y_eval)
                Y_predicted = (self.nn.predict(X_eval) >= 0.5) + 0
                eval_f1 = f1_score(Y_predicted, Y_eval)
                eval_correct_rate = numpy.sum(Y_predicted == Y_eval) / Y_eval.shape[1]
            
            if epoch > 10:
                loss_metric = max(collective_loss, eval_loss)
                self.lr_scheduler.step(loss_metric)
                # Stop when learning rate reaches the bottom
                if self.lr_scheduler.base_lr == self.lr_scheduler.min_lr:
                    print(f'Automatically stopping at epoch {epoch}.')
                    break
            
            if should_log:
                print(f'Epoch {epoch}: Train loss {"%.9f" % collective_loss}, eval loss {"%.9f" % eval_loss}, eval f1-score {"%.4f" % eval_f1}, eval correct rate {"%.4f" % eval_correct_rate}, lr {"%.6f" % lr}')
            
            
        os.makedirs('model/', exist_ok=True)
        pickle.dump(self.nn.dump_params(), open('model/model.pickle', 'wb'))
        print('Model saved to model/model.pickle')
