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
        self.best_f1 = -1
        self.wait = 0

    def step(self, current_f1):
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
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
        self.best_eval_loss = float('inf')  # 记录最佳验证集损失
        self.patience = 10  # 允许的连续无提升 epoch 数
        self.wait = 0  # 当前连续无提升 epoch 数
        self.lr_scheduler = ReduceLROnPlateau(base_lr=5, factor=0.1, patience=5, min_lr=1e-5)
    def train(self, X_train, Y_train, X_eval, Y_eval):
        epoch = -1
        #def lr(epoch):
            #return base_lr / 2 ** (epoch / 60)  # Scheduled learning rate, decays over time
        
        #eval_positive_rate = numpy.sum(Y_eval) / Y_eval.size
        #print(f'INFO: Evaluation set positive rate is {"%.4f" % eval_positive_rate}')
        
        termination_flag_msg = 'Make change to this file and save to terminate training.'
        os.makedirs('trace/', exist_ok=True)
        open('trace/termination_flag', 'w', encoding='utf-8').write(termination_flag_msg + '\n')
        print('Ready to start. The training with proceed INDEFINITELY.')
        print('Edit the contents in trace/termination_flag to properly stop training.')
        time.sleep(2)
        
        while True:
            epoch += 1
            should_log = (epoch % 2 == 0)
            
            collective_loss = 0
            
            if epoch > 0:
                if termination_flag_msg != open('trace/termination_flag', 'r', encoding='utf-8', errors='ignore').read().strip():
                    print('Termination flag change detected. Terminating...')
                    break
                
                train_data = numpy.vstack([ X_train, Y_train ]).transpose()
                numpy.random.shuffle(train_data)
                n_batches = train_data.shape[0] // 500 + 1  # Batch size ~
                batches: list[numpy.ndarray] = numpy.array_split(train_data, n_batches, axis=0)
                
                for batch in batches:
                    if batch.shape[0] == 0:
                        continue
                    batch = batch.transpose()
                    batch_X = batch[:69]
                    batch_Y = batch[69:70]
                    collective_loss += batch_X.shape[1] * self.nn.train(batch_X, batch_Y, self.lr_scheduler.base_lr, epoch)
                    
                collective_loss /= train_data.shape[0]
            
            eval_loss = numpy.nan
            eval_f1 = numpy.nan
            eval_correct_rate = numpy.nan
            if X_eval is not None:
                eval_loss = self.nn.eval(X_eval, Y_eval)
                Y_predicted = (self.nn.predict(X_eval) >= 0.5) + 0
                eval_f1 = f1_score(Y_predicted, Y_eval)
                eval_correct_rate = numpy.sum(Y_predicted == Y_eval) / Y_eval.shape[1]

                self.lr_scheduler.step(eval_f1)
            # 早停逻辑
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.wait = 0
                else:
                    self.wait += 1
                    # if self.wait >= self.patience:
                    #     print(f'Early stopping at epoch {epoch}.')
                    #     break
                    
            if should_log:
                print(f'Epoch {epoch}: Train loss {"%.9f" % collective_loss}, eval loss {"%.9f" % eval_loss}, eval f1-score {"%.4f" % eval_f1}, eval correct rate {"%.4f" % eval_correct_rate}')
            
            
        os.makedirs('model/', exist_ok=True)
        pickle.dump(self.nn.dump_params(), open('model/model.pickle', 'wb'))
        print('Model saved to model/model.pickle')
