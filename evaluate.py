import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from model import Model
from dataset import SingleFileTestDataset
from torch.utils.data import DataLoader
import time
import sys
import os
import numpy as np

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
num_epochs = 200
learning_rate = 1e-4
num_layers = 1
model_dim = 320
num_heads = 4
ff_dim = 320
random_seed = 69
batch_size = int(sys.argv[3])
dropout = 0.3
loss_function = torch.nn.BCELoss()
model_name = sys.argv[1]
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

# Set is_eval to True to keep ESM module on CPU (saves memory?)
my_model = Model(device, 1, 320, 4, 320, 0.3, is_eval=True)
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device(device)))
print(my_model)
my_model.eval()

model_time = time.time()
print(f'Model init: {model_time - start_time}')

print('=====================DATA======================')

data_file = dir_path + '/data/' + sys.argv[2]
print(data_file)
test_set = SingleFileTestDataset(data_file, threshold=int(sys.argv[4]))
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print(len(test_set))

data_time = time.time()
print(f'Data load: {data_time - model_time}')

print('=====================EVALUATION======================')

logfile = f'{dir_path}/output/{sys.argv[5]}.csv'
print(logfile)

y_score = []
y_true = []
correct = 0
total = 0

with torch.no_grad():
    
    for data in iter(test_loader):

        loop_benchmarks = []
        loop_benchmarks.append(time.time())
        
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))

        loop_benchmarks.append(time.time())

        y_true.extend(y.tolist())
        outputs = my_model(inputs).squeeze().detach().cpu()

        loop_benchmarks.append(time.time())

        total += len(inputs)
        correct += (abs(outputs - y) < 0.5).sum().item()

        if len(inputs) > 1:
            y_score.extend(outputs.tolist())
        else:
            y_score.append(outputs.item())
        
        # print(loop_benchmarks[1] - loop_benchmarks[0], loop_benchmarks[2] - loop_benchmarks[1])

print(y_score, len(y_score))
print(y_true, len(y_true))
with open(logfile, 'w') as outfile:
    outfile.write('scores,labels\n')
    for i in range(len(y_score)):
        outfile.write(f'{y_score[i]},{y_true[i]}\n')

print('=====================METRICS======================')

y_score = np.array(y_score)
y_true = np.array(y_true)

print(f'Accuracy: {correct/total}. {correct}/{total}')

auroc = sklearn.metrics.roc_auc_score(y_true, y_score)
print(f'AUROC: {auroc}')
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
plt.figure(0)
plt.plot(fpr, tpr)
plt.text(0.5, 0.5, auroc)
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
plt.text(0.5, 0.5, auprc)
plt.title(f'{model_name} PRC')
plt.xlabel('Recall')
plt.ylabel('Precision')
plot_f = f'llps-v2/figures/{model_name}_prc.png'
plt.savefig(fname=plot_f)
print(f'saved to {plot_f}')

f1 = sklearn.metrics.f1_score(y_true, y_score.round())
print(f'F1 Score: {f1}')

end_time = time.time()

print(f'Elapsed: {end_time - start_time}')
