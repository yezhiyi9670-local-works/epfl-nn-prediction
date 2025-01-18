import numpy as np
import os
import argparse

from MyNeuralNetwork import MyNeuralNetwork
from TrainingController import TrainingController
from preprocessing import preprocess_input

class Your_model_name():
    def __init__(self) -> None:
        pass

class dataset():
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def data_collection(self):
        '''
            Parameter
            ---------
            data_path: the input data path, which can be a folder or a npy file 
            
            return
            ------
            x: the training features, with shape of (N, 69), N is the sample size
            y: the training lables, with shape of (N, 1), N is the sample size
        '''
        def normalization(x):
            def normalize(vector):
                max_vals = np.max(vector, axis=0)
                min_vals = np.min(vector, axis=0)
                normalized_vector = (vector - min_vals) / \
                    (max_vals - min_vals + 1e-3)
                return normalized_vector
            if not np.all(x<=1):
                x = normalize(x)
            return x
        try:
            if self.data_path.endswith('npy'):
                data = np.load(self.data_path, allow_pickle=True).item()
                x = normalization(data['features_list'][0])
                y = data['labels_list'][0]
            else:
                x, y = [], []  # Initialize lists to store features and labels
                for npy_file in os.listdir(self.data_path):
                    npy_file_path = os.path.join(self.data_path, npy_file)
                    data = np.load(npy_file_path, allow_pickle=True).item()
                    x.append(normalization(data['features_list'][0]))
                    y.append(data['labels_list'][0])
                x = np.vstack(x).astype(np.float32) 
                y = np.vstack(y).astype(np.float32).reshape(-1)

            return x, y

        except Exception as e:
            assert False, f"An error occurred while loading data: {e}"  # Ensures correct return type

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The training process')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--eval_path', default='', type=str)
    args = parser.parse_args()
    
    data = dataset(args.data_path)
    training_x, training_y = data.data_collection()
    training_x = training_x.transpose()
    training_y = training_y.reshape(1, -1)
    
    print(f'the training features of the circuits are: {training_x} with shape of {training_x.shape}')
    print(f'the training labels of the circuits are: {training_y} with shape of {training_y.shape}')
    
    has_eval = (args.eval_path != '')
    eval_x, eval_y = None, None
    if has_eval:
        eval_set = dataset(args.eval_path)
        eval_x, eval_y = eval_set.data_collection()
        eval_x = eval_x.transpose()
        eval_y = eval_y.reshape(1, -1)
        
        print(f'the evaluating features of the circuits are: {eval_x} with shape of {eval_x.shape}')
        print(f'the evaluating labels of the circuits are: {eval_y} with shape of {eval_y.shape}')
    
    # your implementation
    '''
        After collecting the training dataset, you have to train your classificaiton model
        for example:
            model = Your_model_name(training_x, training_y, other parameters)
            model.training()
        The requirements for the model.training() should include:
            1. the detailed training process
            2. the model performance (recall, precision, f1_score) on the training dataset
            3. save the final model
    '''
    
    nn = MyNeuralNetwork()
    controller = TrainingController(nn)
    
    # Void the node ID column
    training_x = preprocess_input(training_x)
    if eval_x is not None: eval_x = preprocess_input(eval_x)
    
    if has_eval:
        print('Training with hot evaluation.')
    else:
        print('Training without hot evaluation.')
    controller.train(training_x, training_y, eval_x, eval_y)

