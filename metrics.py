import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt

model_name = '2022-12-21_full_e200_lr-4_dropout-0.3'
df = pd.read_csv(f'llps-v2/output/testlog.csv')
# df_pos = pd.read_csv('llps-v2/data/ext_pos.csv')
# df_pos.columns = ['sequences']
# df_neg = pd.read_csv('llps-v2/data/ext_neg.csv')
# df_neg.columns = ['sequences']
# df_seqs = pd.concat([df_pos, df_neg]).reset_index(drop=True)
# df_seqs = df_seqs.iloc[0:319]
# df = pd.concat([df, df_seqs], axis=1)
# print(df)
# df2 = df[df.iloc[:,2].str.len() <= 2000].reset_index(drop=True)
# print(df2)

# y_score = df2['scores']
# y_true = df2['labels']

y_score = df['scores']
y_true = df['labels']

correct = 0
total = 0
for i in range(len(y_score)):
    total += 1
    if abs(y_score[i] - y_true[i]) < 0.5:
        correct += 1 

print('=====================METRICS======================')

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