import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from model import Model
from dataset import ProteomeDataset
from torch.utils.data import DataLoader
import time
import sys
import os

start_time = time.time()

dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

print('=====================INPUTS===========================')

print(f'model_name: {sys.argv[1]}')
print(f'dataset: {sys.argv[2]}')
print(f'batch_size: {sys.argv[3]}')
print(f'threshold: {sys.argv[4]}')
print(f'logfile: {sys.argv[5]}')

print('=====================HYPERPARAMS======================')
num_layers = 1
model_dim = 320
num_heads = 4
ff_dim = 320
random_seed = 69
batch_size = int(sys.argv[3])
dropout = 0.3
loss_function = torch.nn.BCELoss()
model_name = sys.argv[1]
path = dir_path + '/sample-model/'

print(f'num_layers: {num_layers}')
print(f'model_dim: {model_dim}')
print(f'num_heads: {num_heads}')
print(f'ff_dim: {ff_dim}')
print(f'batch_size: {batch_size}')
print(f'dropout: {dropout}')
print(f'model_name: {model_name}')
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

data_file = dir_path + '/../proteome_evaluation/proteome_batches/'+sys.argv[2]
print(data_file)
data_set = ProteomeDataset(data_file, 'sequence', 'uniprot_id', -1)
data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
print(len(data_set))

data_time = time.time()
print(f'Data load: {data_time - model_time}')

print('=====================PREDICTIONS======================')

logfile = f'{dir_path}/../proteome_evaluation/model_predictions/{sys.argv[5]}.csv'
print(logfile)

with open(logfile, 'w') as outfile:
    outfile.write('scores,labels\n') 

with torch.no_grad():
    with open(logfile, 'a') as outfile:

        for data in iter(data_loader):
        
            seq = data[0][0]
            label = data[1][0]
            input = [(label, seq)]

            output = my_model(input).squeeze().detach().cpu().item()
            print(label, output)
            outfile.write(f'{output},{label}\n')

end_time = time.time()
print(end_time - start_time)