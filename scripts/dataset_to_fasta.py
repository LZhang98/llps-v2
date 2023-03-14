import csv

file = 'llps-v2/data/ext_neg.csv'
output = 'llps-v2/data/ext_neg.fa'

seqs = []
labels = []
with open(file) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        seqs.append(row[0])

outfile = open(output, 'w')
for i in range(len(seqs)):
    label = 'seq'+str(i)
    outfile.write('>' + label + '\n')
    outfile.write(seqs[i] + '\n')
