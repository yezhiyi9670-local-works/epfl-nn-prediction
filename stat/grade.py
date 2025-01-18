import numpy as np
import argparse
import re

import subprocess

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    if not process.stdout:
        return ''
    total_output = ''
    while True:
        poll_result = process.poll()
        output = process.stdout.readline().decode('utf-8', errors='ignore')
        if output == '' and poll_result is not None:
            break
        if output:
            print(output, end='')
            total_output += output
    return total_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The training process')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--eval_path', default='', type=str)
    args = parser.parse_args()
    
    data_path = args.data_path
    eval_path = args.eval_path
    
    train_command = [ 'python', 'main.py', '--data_path', data_path ]
    eval_command = [ 'python', 'For_TA_test.py', '--data_path', eval_path ]
    
    min_score = np.inf
    max_score = -np.inf
    avg_score = 0
    
    rounds = 5
    
    for i in range(rounds):
        run_command(train_command)
        eval_output = run_command(eval_command)
        
        matched = re.search(r"f1_score:\s([\d.]+)", eval_output)
        
        if not matched:
            print('The evaluation has failed!')
            exit(-1)

        f1_score = float(matched.group(1))
        min_score = min(min_score, f1_score)
        max_score = max(max_score, f1_score)
        avg_score += f1_score / rounds
    
    print(f'Finished. min: {"%.4f" % min_score}, avg: {"%.4f" % avg_score}, max: {"%.4f" % max_score}')
