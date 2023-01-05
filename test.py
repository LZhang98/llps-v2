import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from model import Model
from dataset import SingleFileTestDataset, SingleFileDataset
from torch.utils.data import DataLoader
import time
import sys

start_time = time.time()

print('=====================INPUTS========================')

for i in range(len(sys.argv)):
    print(f'{i} {sys.argv[i]}')

sys.exit()

print('=====================HYPERPARAMS======================')
num_epochs = 200
learning_rate = 1e-4
num_layers = 1
model_dim = 320
num_heads = 4
ff_dim = 320
random_seed = 69
batch_size = 5
dropout = 0.3
loss_function = torch.nn.BCELoss()
model_name = '2022-12-15_full_e200_lr-4_dropout-0.3'
path = '/cluster/projects/kumargroup/luke/output/v2/'

print(f'model_name: {model_name}')
torch.manual_seed(random_seed)

print('=====================MODEL======================')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3)
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt'))
print(my_model)
my_model.eval()

print('=====================DATA======================')

data = SingleFileDataset('llps-v2/data/training_data_features.csv')

dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

test = next(iter(dataloader))
print(test)

end_time = time.time()

print(f'Elapsed: {end_time - start_time}')