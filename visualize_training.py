import torch
from torch.utils.data import DataLoader
import dataset
import numpy as np
from model import Model
import csv
import matplotlib.pyplot as plt

model_name = '2023-02-02_e100_bs4'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data = dataset.SingleFileDataset('llps-v2/data/toy_dataset/fifteen_unbalanced.csv')
# data = dataset.SingleFileDataset('llps-v2/data/training_data_features.csv')
# dataloader = DataLoader(data, batch_size=8, shuffle=False)

# print('===========MODEL===========')
# my_model = Model(device, 1, 320, 4, 320, 0.3)
# path = '/cluster/projects/kumargroup/luke/output/v2/'
# my_model.load_state_dict(torch.load(f'{path}{model_name}.pt'))
# print(my_model)

# print('===========EVALUATION===========')
# my_model.eval()
# with torch.no_grad():
#     for i, data in enumerate(dataloader):
#         x, y = data
#         inputs = []
#         for n in range(len(x)):
#             inputs.append((y[n], x[n]))
            
#         targets = y.unsqueeze(1).float().to(device)
#         outputs = my_model(inputs)

#         print(outputs)
#         print(y)

print('===========VISUALIZATION===========')
logfile = 'llps-v2/logs/2023-02-02-log.csv'
print(f'opening {logfile}')

e = []
loss = []
pos = []
neg = []

with open(logfile) as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        e.append(float(row[0]))
        loss.append(float(row[1]))
        pos.append(float(row[2]))
        neg.append(float(row[3]))
# print(e)
# print(loss)
# print(pos)
# print(neg)

plt.figure(0)
plt.plot(e, pos, label='0')
plt.plot(e, neg, label='1')
plt.title(f'{model_name} Class Separation (by Mean Score)')
plt.savefig(f'./llps-v2/figures/{model_name}_pred.png')

plt.figure(1)
plt.plot(e, loss)
plt.title(f'{model_name} Training Loss')
plt.savefig(f'./llps-v2/figures/{model_name}_loss.png')
