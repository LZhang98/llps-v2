import pandas as pd

pos = pd.read_csv('llps-v2/data/ext_pos.csv')
neg = pd.read_csv('llps-v2/data/ext_neg.csv')

pos.columns = ['sequences']
neg.columns = ['sequences']

pos['labels'] = 1
neg['labels'] = 0

data = pd.concat([pos, neg])
print(data)
data.to_csv('llps-v2/data/test_set_1_pos.csv')