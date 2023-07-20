import torch
from torch.utils.data import DataLoader
import src.dataset as dataset
from src.model import SimplifiedModel
from datetime import date
import time
import sys
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import os
import json

if __name__ == '__main__':
    
    start_time = time.time()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    print('=====================CONFIGS=====================')

    # print relevant config settings for this script:
    config = json.load(open(dir_path + '/config.json'))
    print(config['training_log_location'])
    print(config['evaluation_dir'])

    training_log_location = config['training_log_location']
    model_save_loc = config['model_save_location']

    print('=====================INPUTS======================')
    
    print(sys.argv)
    if len(sys.argv) < 7:
        print('param error. exiting.')
        sys.exit()

    print(f'num_epochs: {sys.argv[1]}')
    print(f'learning_rate: {sys.argv[2]}')
    print(f'batch_size: {sys.argv[3]}')
    print(f'dropout: {sys.argv[4]}')
    print(f'num_heads: {sys.argv[5]}')
    print(f'training_file: {sys.argv[6]}')

    if len(sys.argv) == 8:
        print(f'tag: {sys.argv[7]}')
        tag = sys.argv[7]
    else:
        print('tag: -')
        tag = ''

    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)


    # Hyperparams
    print('=====================HYPERPARAMS======================')
    num_epochs = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    num_layers = 1
    model_dim = 320
    num_heads = int(sys.argv[5])

    random_seed = 69
    torch.manual_seed(random_seed)

    batch_size = int(sys.argv[3])
    dropout = float(sys.argv[4])
    loss_function = torch.nn.BCELoss()

    today = str(date.today())
    print(today)
    model_name = f'{today}_e{num_epochs}_bs{batch_size}_nh{num_heads}_mhsa_{tag}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'num_epochs: {num_epochs}')
    print(f'learning_rate: {learning_rate}')
    print(f'num_layers: {num_layers}')
    print(f'model_dim: {model_dim}')
    print(f'num_heads: {num_heads}')
    print(f'batch_size: {batch_size}')
    print(f'model_name: {model_name}')
    print(f'dropout: {dropout}')
    print(f'random_seed: {random_seed}')

    # LOGGING 
    logfile = f'{training_log_location}{model_name}_log.csv'
    print(logfile)
    print(f'logfile: {logfile}')
    with open(logfile, 'w') as f:
        f.write('epoch,training_loss,mean_pos,mean_neg,val_loss\n')

    # DATASET
    # toy dataset first
    print('===========DATA===========')
    training_file = dir_path + '/' + sys.argv[6]
    validation_split = 0.2
    data = dataset.SingleFileDataset(training_file, threshold=1500)
    print(training_file)
    dataset_size = len(data)
    print(f'dataset size: {dataset_size}')
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    print(f'training split: {len(train_indices)}; validation split: {len(val_indices)}')

    # Initialize samplers on splits:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Initialize data loaders:
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=1, sampler=val_sampler)

    # MODEL AND OPTIMIZER
    print('===========MODEL===========')
    my_model = SimplifiedModel(device, model_dim, num_heads, dropout)
    print(my_model)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    print('===========TRAINING===========')

    epoch_arr = []
    pos_arr = []
    neg_arr = []
    loss_arr = []
    for epoch in range(num_epochs):

        # TRAINING

        curr_pos = torch.tensor([])
        curr_neg = torch.tensor([])        
        # print(f'starting epoch {epoch}')
        my_model.train()
        running_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data
            inputs = []
            for n in range(len(x)):
                inputs.append((y[n], x[n]))
            
            if batch_size > 1:
                targets = y.unsqueeze(1).float().to(device)
            else:
                targets = y.float().to(device)

            optimizer.zero_grad()
            outputs = my_model(inputs)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            pos_scores = outputs[targets == 1].detach().squeeze().cpu()
            neg_scores = outputs[targets == 0].detach().squeeze().cpu()

            curr_pos = torch.hstack((curr_pos, pos_scores))
            curr_neg = torch.hstack((curr_neg, neg_scores))

            batch_loss = loss.item()
            running_loss += batch_loss * len(inputs)
        
        epoch_arr.append(epoch)
        loss_arr.append(running_loss)
        pos = torch.mean(curr_pos).item()
        neg = torch.mean(curr_neg).item()
        pos_arr.append(pos)
        neg_arr.append(neg)        
    
        # VALIDATION
        my_model.eval()
        with torch.no_grad():
            val_loss = 0
            for i, data in enumerate(val_loader):
                x, y = data
                input = [(y.item(), x[0])]
                target = y.float().to(device)

                output = my_model(input)

                loss = loss_function(output, target)
                val_loss += loss

        print(f'{epoch}, {running_loss}, {pos}, {neg}, {val_loss}')
        with open(logfile, 'a') as f:
            f.write(f'{epoch}, {running_loss}, {pos}, {neg}, {val_loss}\n')
    
    # SAVE THE MODEL
    print('===========SAVING===========')

    path = model_save_loc + model_name + '.pt'
    torch.save(my_model.state_dict(), path)
    print(f'Save to {path}')

    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)