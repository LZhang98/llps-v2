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
print(test_lst)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_model = Model(device, 1, 320, 4, 320, 0.3, False, is_eval=True)
path = config.model['model_save_location']
proj_location = config.PROJECT_LOCATION
my_model.load_state_dict(torch.load(f'{path}{model_name}.pt', map_location=torch.device(device)))

num_samples = len(data)

# Test predictions

test_output = my_model.predict(test_lst)
print(test_output)

# Set up interpreter

LIME_explainer = lime_text.LimeTextExplainer()

explanations = []
for i in range(5):
    exp = LIME_explainer.explain_instance(test_lst[i], my_model.predict, num_features=50)
    explanations.append(exp)

for i in range(5):
    exp = explanations[i]
    exp.as_pyplot_figure()
    plt.savefig(output_dir + f'test_{i}.png')