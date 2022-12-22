import csv

file = 'llps-v2/data/toy_dataset/fifteen_unbalanced.csv'
output = 'llps-v2/data/toy_dataset/fifteen_unbalanced.fa'

seqs = []
labels = []
with open(file) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        seqs.append(row[0])
        labels.append(row[1])

outfile = open(output, 'w')
for i in range(len(seqs)):
    print(labels[i])
    outfile.write('>' + labels[i] + '\n')
    outfile.write(seqs[i] + '\n')
