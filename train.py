import torch
from torch.utils.data import DataLoader
import dataset
import numpy as np
from model import Model
from datetime import date
from esm_pretrained import ESM
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    
    start_time = time.time()

    # Hyperparams
    print('=====================HYPERPARAMS======================')
    num_epochs = 200
    learning_rate = 1e-4
    num_layers = 1
    model_dim = 320
    num_heads = 4
    ff_dim = 320
    random_seed = 69
    batch_size = 8
    dropout = 0.3
    loss_function = torch.nn.BCELoss()

    today = str(date.today())
    print(today)
    model_name = f'{today}_full_e{num_epochs}_lr{int(np.log10(learning_rate))}_dropout-{dropout}_2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'num_epochs: {num_epochs}')
    print(f'learning_rate: {learning_rate}')
    print(f'num_layers: {num_layers}')
    print(f'model_dim: {model_dim}')
    print(f'num_heads: {num_heads}')
    print(f'ff_dim: {ff_dim}')
    print(f'batch_size: {batch_size}')
    print(f'model_name: {model_name}')
    print(f'dropout: {dropout}')
    torch.manual_seed(random_seed)

    # LOGGING 
    logfile = f'{model_name}_log.csv'
    print(f'logfile: {logfile}')

    # DATASET
    # toy dataset first
    print('===========DATA===========')
    # data = dataset.SingleFileDataset('llps-v2/data/toy_dataset/ten_balanced.csv')
    data = dataset.SingleFileDataset('llps-v2/data/training_data_features.csv')

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)

    # MODEL AND OPTIMIZER
    print('===========MODEL===========')
    my_esm = ESM(320)
    my_model = Model(device, num_layers, model_dim, num_heads, ff_dim, dropout)
    print(my_model)
    # my_model.to(device)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate)

    print('===========TRAINING===========')
    # TRAINING LOOP
    epoch_arr = []
    pos_arr = []
    neg_arr = []
    loss_arr = []
    for epoch in range(num_epochs):

        curr_pos = torch.tensor([])
        curr_neg = torch.tensor([])        
        # print(f'starting epoch {epoch}')
        my_model.train()
        running_loss = 0
        for i, data in enumerate(dataloader):
            x, y = data
            inputs = []
            for n in range(len(x)):
                inputs.append((y[n], x[n]))
            
            targets = y.unsqueeze(1).float().to(device)

            optimizer.zero_grad()
            outputs = my_model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            pos_scores = outputs[targets == 0].detach().squeeze().cpu()
            neg_scores = outputs[targets == 1].detach().squeeze().cpu()

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
        print(f'{epoch}, {running_loss}, {pos}, {neg}')
    
    # SAVE THE MODEL
    print('===========SAVING===========')

    path = '/cluster/projects/kumargroup/luke/output/v2/' + model_name + '.pt'
    torch.save(my_model.state_dict(), path)
    print(f'Save to {path}')

    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)

    print('===========EVALUATION===========')

    # my_model.eval()
    # with torch.no_grad():
    #     prediction = my_model(test_batch)
    #     print(prediction)

    # plt.figure(0)
    # plt.plot(epoch_arr, pos_arr, label='0')
    # plt.plot(epoch_arr, neg_arr, label='1')
    # plt.savefig(f'./llps-v2/figures/{model_name}_pred.png')
    
    # plt.figure(1)
    # plt.plot(epoch_arr, loss_arr)
    # plt.savefig(f'./llps-v2/figures/{model_name}_loss.png')
