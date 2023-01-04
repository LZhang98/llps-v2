import pandas as pd

df = pd.read_csv('llps-v2/data/test_set_1_pos.csv')

seqs = df['sequences']

total = len(seqs)
count = 0
for seq in seqs:
    if len(seq) > 2000:
        count += 1

print(count)
print(count/total)