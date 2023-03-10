import torch
from model import Model
from datetime import date
import time
import sys
import pandas as pd

start_time = time.time()

# Script input args
model_name = sys.argv[1]
data_file = sys.argv[2]
if len(sys.argv) == 4:
    output_dir = sys.argv[3]
else:
    output_dir = 'llps-v2/predictions/'

# Set up data input and output
output_file = f'{output_dir}/{model_name}_predictions2500.csv'
with open(output_file, 'w') as f:
    f.write('Score,Uniprot ID,Type,Length,Sequence\n')

data = pd.read_csv(data_file)
data['Length'] = data['Protein Sequence'].str.len()

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3, False, is_eval=True)
path = '/cluster/projects/kumargroup/luke/output/v2/'
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device(device)))

num_samples = len(data)
with torch.no_grad():
    for i in range(num_samples):
        seq = data.at[i, 'Protein Sequence']
        id = data.at[i, 'UniProt ID']
        type = data.at[i,'LLPS Type']
        length = data.at[i,'Length']

        if (length < 2500):
            input = [(id, seq)]

            output = my_model(input).squeeze().detach()

            result = f'{output},{id},{type},{length},{seq}\n'

            with open(output_file, 'a') as f:
                f.write(result)

end_time = time.time()

print(f'Elapsed: {end_time - start_time}')