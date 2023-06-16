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
if 'type' not in data.columns:
    data['type'] = pd.NA

print('==============I/O=================')
print(f'output: {output_file}')
print(f'input: {data_file}')
print(f'dataset size: {len(data.index)}')

print('=============MODEL================')

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3, False, is_eval=True)
path = config.model['model_save_location']
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device(device)))

num_samples = len(data)
with torch.no_grad():
    for i in range(num_samples):
        seq = data.at[i, 'sequence']
        id = data.at[i, 'id']
        type = data.at[i,'type']
        length = data.at[i,'length']

        if (length < 2500):
            input = [(id, seq)]

            output = my_model(input).squeeze().detach()

            result = f'{output},{id},{type},{length},{seq}\n'

            # with open(output_file, 'a') as f:
            f.write(result)

end_time = time.time()

print(f'Elapsed: {end_time - start_time}')