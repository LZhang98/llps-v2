import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from model import Model
from dataset import SingleFileTestDataset
from torch.utils.data import DataLoader

print('=====================HYPERPARAMS======================')
num_epochs = 200
learning_rate = 1e-4
num_layers = 1
model_dim = 320
num_heads = 4
ff_dim = 320
random_seed = 69
batch_size = 8
dropout = 0.3
loss_function = torch.nn.BCELoss()
model_name = '2022-12-15_full_e200_lr-4_dropout-0.3'
path = '/cluster/projects/kumargroup/luke/output/v2/'

print(f'num_epochs: {num_epochs}')
print(f'learning_rate: {learning_rate}')
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
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt'))
print(my_model)
my_model.eval()

print('=====================DATA======================')

data_dir = 'llps-v2/data/'
print(f'pos_file: {data_dir} + ext_pos.csv')
print(f'neg_file: {data_dir} + ext_neg.csv')
pos_data = SingleFileTestDataset(data_dir + 'ext_pos.csv', 0)
neg_data = SingleFileTestDataset(data_dir + 'ext_neg.csv', 1)

pos_loader = DataLoader(pos_data, batch_size=batch_size, shuffle=False)
neg_loader = DataLoader(neg_data, batch_size=batch_size, shuffle=False)

print('=====================EVALUATION======================')

y_score = torch.tensor([])
y_true = torch.tensor([])
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(pos_loader):
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))
        
        y_true = torch.hstack(y_true, y)
        outputs = my_model(inputs).squeeze().cpu()

        total += len(inputs)
        correct += (abs(y_score - y_true) < 0.5).sum().item()

        y_score = torch.hstack(y_score, outputs)
    
    for i, data in enumerate(neg_loader):
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))
        
        y_true = torch.hstack(y_true, y)
        outputs = my_model(inputs)

        y_score = torch.hstack(y_score, outputs)

print(y_score)
print(y_true)

print('=====================METRICS======================')

print(f'Accuracy: {correct/total}. {correct}/{total}')

auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
print(f'AUROC: {auroc}')
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
plt.figure(0)
plt.plot(fpr, tpr)
plt.title(f'{model_name} ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plot_f = f'llps-v2/figures/{model_name}_roc.png'
plt.savefig(fname=plot_f)
print(f'saved to {plot_f}')

auprc = sklearn.metrics.average_precision_score(y_true, y_score)
print(f'AUPRC: {auprc}')
precision, recall, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
plt.figure(1)
plt.plot(recall, precision)
plt.title(f'{model_name} PRC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plot_f = f'llps-v2/figures/{model_name}_prc.png'
plt.savefig(fname=plot_f)
print(f'saved to {plot_f}')