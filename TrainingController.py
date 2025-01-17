from MyNeuralNetwork import MyNeuralNetwork

import time
import os
import numpy
import nn_modules
import random
import pickle
from formula import f1_score

class TrainingController():
    def __init__(self, nn: MyNeuralNetwork):
        self.nn = nn
    def train(self, X_train, Y_train, X_eval, Y_eval):
        epoch = -1
        base_lr = 10
        batch_size = 5000
        def lr(epoch):
            return base_lr / 2 ** max(0, (epoch - 0) / 60)  # Scheduled learning rate, decays over time
        
        eval_positive_rate = numpy.sum(Y_eval) / Y_eval.size
        print(f'INFO: Evaluation set positive rate is {"%.4f" % eval_positive_rate}')
        
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
                n_batches = train_data.shape[0] // batch_size + 1  # Batch size ~
                batches: list[numpy.ndarray] = numpy.array_split(train_data, n_batches, axis=0)
                
                for batch in batches:
                    if batch.shape[0] == 0:
                        continue
                    batch = batch.transpose()
                    batch_X = batch[:69]
                    batch_Y = batch[69:70]
                    collective_loss += batch_X.shape[1] * self.nn.train(batch_X, batch_Y, lr(epoch), epoch)
                    
                collective_loss /= train_data.shape[0]
            
            eval_loss = numpy.nan
            eval_f1 = numpy.nan
            eval_correct_rate = numpy.nan
            if X_eval is not None:
                eval_loss = self.nn.eval(X_eval, Y_eval)
                Y_predicted = (self.nn.predict(X_eval) >= 0.5) + 0
                eval_f1 = f1_score(Y_predicted, Y_eval)
                eval_correct_rate = numpy.sum(Y_predicted == Y_eval) / Y_eval.shape[1]
            
            if should_log:
                print(f'Epoch {epoch}: Train loss {"%.9f" % collective_loss}, eval loss {"%.9f" % eval_loss}, eval f1-score {"%.4f" % eval_f1}, eval correct rate {"%.4f" % eval_correct_rate}')
            
            
        os.makedirs('model/', exist_ok=True)
        pickle.dump(self.nn.dump_params(), open('model/model.pickle', 'wb'))
        print('Model saved to model/model.pickle')
