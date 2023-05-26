import torch
import sklearn.metrics
import matplotlib.pyplot as plt
from model import Model
from dataset import SingleFileTestDataset, SingleFileDataset
from torch.utils.data import DataLoader
import time
import sys
import config

print(config.model['model_dir'])
print(config.model['output_dir'])