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

model_name = '2023-01-04_full_e200_lr-4_dropout-0.3'
path = '/cluster/projects/kumargroup/luke/output/v2/'

print(f'model_name: {model_name}')
torch.manual_seed(random_seed)

print('=====================MODEL======================')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3)
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device('cpu')))
print(my_model)
my_model.eval()

activation = {}
def get_activation(name):
    def hook(model, input, output):
        if name in activation:
            curr = activation[name]
            activation[name] = torch.cat((curr, input[0]), 0)
        else:
            activation[name] = input[0]
    return hook

my_model.encoder.register_forward_hook(get_activation('encoder'))
my_model.classifier.register_forward_hook(get_activation('classifier'))

print('=====================DATA======================')

dataset = SingleFileTestDataset('llps-v2/data/test_8_short_seqs.csv')

print('================BATCH SIZE 1======================')

dataloader1 = DataLoader(dataset, batch_size=1, shuffle=False)

outputs1 = torch.tensor([])
for data in iter(dataloader1):
    with torch.no_grad():
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))
        out = my_model(inputs).squeeze().detach().cpu()
        print(out)

        torch.cat((outputs1, out), dim=0)

embeds1 = activation['encoder']
print(embeds1.size())
features1 = activation['classifier']
print(features1.size())
print(outputs1)
activation = {}

print('================BATCH SIZE 5======================')

outputs5 = torch.tensor([])
dataloader5 = DataLoader(dataset, batch_size=5, shuffle=False)
for data in iter(dataloader5):
    with torch.no_grad():
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))

        torch.cat((outputs5, my_model(inputs).squeeze().detach().cpu()), dim=0)

embeds5 = activation['encoder']
print(embeds5.size())
features5 = activation['classifier']
print(features5.size())
print(outputs5)
activation = {}

print('================BATCH SIZE 8======================')

outputs8 = torch.tensor([])
dataloader8 = DataLoader(dataset, batch_size=8, shuffle=False)
for data in iter(dataloader8):
    with torch.no_grad():
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))

        torch.cat((outputs8, my_model(inputs).squeeze().detach().cpu()), dim=0)

embeds8 = activation['encoder']
print(embeds8.size())
features8 = activation['classifier']
print(features8.size())
print(outputs8)
activation = {}


end_time = time.time()

print(f'Elapsed: {end_time - start_time}')