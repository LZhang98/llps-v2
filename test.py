import torch
from model import Model
import time
import sys
import pandas as pd
import config

start_time = time.time()

# Script input args
model_name = sys.argv[1]
data_file = sys.argv[2]
output_name = sys.argv[3]
output_dir = config.model['output_dir']

print('===============INPUTS================')

print(model_name, data_file, output_name, output_dir)

# Set up data input and output
output_file = f'{output_dir}/{model_name}_{output_name}.csv'
f = open(output_file, 'w')
f.write('score,id,type,length,sequence\n')

data = pd.read_csv(data_file)
data['length'] = data['sequence'].str.len()

print('==============I/O=================')
print(f'output: {output_file}')
print(f'input: {data_file}')
print(f'dataset size: {len(data.index)}')