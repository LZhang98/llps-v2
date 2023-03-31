import torch
from model import Model
import time
import sys
import os
import pandas as pd

start_time = time.time()

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

print('=====================INPUTS===========================')

print(f'model_name: {sys.argv[1]}')
print(f'uniprot_id: {sys.argv[2]}')

print('=====================HYPERPARAMS======================')
num_layers = 1
model_dim = 320
num_heads = 4
ff_dim = 320
random_seed = 69
dropout = 0.3
model_name = sys.argv[1]
uniprot_id = sys.argv[2]
path = dir_path + '/sample-model/'

print(f'model_name: {model_name}')
print(f'uniprot_id: {sys.argv[2]}')
torch.manual_seed(random_seed)

print('=====================MODEL======================')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3)
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device(device)))
print(my_model)
my_model.eval()

model_time = time.time()
print(f'Model init: {model_time - start_time}')

print('=====================DATA======================')

data_file = dir_path + '/../proteome_evaluation/proteome.csv'
print(data_file)
proteome = pd.read_csv(data_file)

target_row = proteome.loc[proteome['uniprot_id'] == uniprot_id]
print(target_row)
sequence = target_row['sequence'].values[0]
print(sequence)

data_time = time.time()
print(f'Data load: {data_time - model_time}')

print('=====================PREDICTIONS======================')

with torch.no_grad():
        
    input = [(uniprot_id, sequence)]

    output = my_model(input).squeeze().detach().cpu().item()
    print(uniprot_id, output)

end_time = time.time()
print(end_time - start_time)