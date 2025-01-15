import numpy as np
import os
import argparse
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
            print(f"An error occurred while loading data: {e}")  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The test process')
    # TA provdie the test data_path
    parser.add_argument('--data_path', default='', type=str)
    # you provide the model_path in default
    parser.add_argument('--model_path', default='your relative model path', type=str)
    args = parser.parse_args()
    data = dataset(args.data_path)
    test_x, test_y = data.data_collection()
    print(f'the test features of the circuits are: {test_x} with shape of {test_x.shape}')
    print(f'the test labels of the circuits are: {test_y} with shape of {test_y.shape}')
    # your implementation
    '''
        The TA only provide the test_data_path, and you have to supplement the code.
        Notice that The TA only provide the test_data_path, 
        and the output of For_TA_test.py should be the recall, precision and f1_score of your model on the test dataset. 
        
        for example:
            model = Your_model_name(test_x, test_y, other parameters)
            model.load(args.model_path)
            recall, precision, f1_score = model.test()
            print(f'the performance of the model is recall: {recall}, precision: {precision}, f1_score: {f1_score}')
        
        The grade will depend on you model's f1_score
    '''

