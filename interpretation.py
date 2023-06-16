import torch
from model import Model
import time
import sys
import pandas as pd
import config
from lime import lime_text
import matplotlib.pyplot as plt

start_time = time.time()

# Script input args
print(sys.argv)
model_name = sys.argv[1]
data_file = sys.argv[2]

output_dir = config.interp['output_dir']

data = pd.read_csv(data_file)

# for testing, take 5 sequences from data file
# TODO: delete after testing

test_lst = data['sequences'][0:5].values.tolist()
print('\nTEST LIST')
for x in test_lst:
    print(len(x), x)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3, False, is_eval=True)
path = config.model['model_save_location']
proj_location = config.PROJECT_LOCATION
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device(device)))

num_samples = len(data)

# Test predictions

print('\nPREDICTIONS')
test_output = my_model.predict(test_lst)
print(test_output)
print(test_output.size())

# Set up interpreter
print('\nINTERPRETATION')
def my_split(seq):
    # return list(seq)
    return seq

LIME_explainer = lime_text.LimeTextExplainer(split_expression=my_split, class_names=['non-LLPS', 'LLPS'], verbose=True, char_level=True)

exp = LIME_explainer.explain_instance(test_lst[0][0:100], my_model.predict, num_features=10)

print(exp.as_list())

print('interpretation done. save fig')

for i in range(1):
    exp.as_pyplot_figure()
    plt.savefig(output_dir + f'test_{i}.png')

end_time = time.time()
duration = end_time - start_time
print(f'elapsed: {duration} seconds or {duration/60} minutes', )