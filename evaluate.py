import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from model import Model
from dataset import SingleFileTestDataset
from torch.utils.data import DataLoader
import time

start_time = time.time()

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
model_name = '2022-12-21_full_e200_lr-4_dropout-0.3'
# model_name = '2022-12-22_random_baseline'
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

data_file = 'llps-v2/data/test_set_1_pos.csv'
print(data_file)
test_set = SingleFileTestDataset(data_file, threshold=2000)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print(len(test_set))

print('=====================EVALUATION======================')

logfile = f'llps-v2/output/{model_name}_eval_log_2000.csv'
print(logfile)

y_score = []
y_true = []
correct = 0
total = 0

with torch.no_grad():
    for data in iter(test_loader):
        x, y = data
        inputs = []
        for n in range(len(x)):
            inputs.append((y[n], x[n]))

        y_true.extend(y.tolist())
        outputs = my_model(inputs).squeeze().detach().cpu()

        total += len(inputs)
        correct += (abs(outputs - y) < 0.5).sum().item()

        if len(inputs) > 1:
            y_score.extend(outputs.tolist())
        else:
            y_score.append(outputs.item())

print(y_score, len(y_score))
print(y_true, len(y_true))
with open(logfile, 'w') as outfile:
    outfile.write('scores,labels\n')
    for i in range(len(y_score)):
        outfile.write(f'{y_score[i]},{y_true[i]}\n')

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

end_time = time.time()

print(f'Elapsed: {end_time - start_time}')