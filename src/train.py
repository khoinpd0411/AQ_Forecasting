import os
import time
from tqdm import tqdm
import copy
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataloaders import Vanilla_LSTMDataset
from model import Vanilla_LSTM

def train_vanilla(model, dataloaders, dataset_sizes,
                criterion, optimizer, scheduler,
                device, epochs, save_path):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e6

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                # hiddens = model.initHidden(inputs.float())

                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print('Outputs: ', outputs)
                    # print('Labels: ', labels.size())
                    loss = criterion(outputs, labels.unsqueeze(-1))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, save_path)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')

    return model

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = './data/data-train', help = 'data dir')
    parser.add_argument('--location_dir', type = str, default = './data/data-train/location', help = 'location dir')
    parser.add_argument('--save_path', type=str, default = './weights/lstm-aq.pth', help='initial weights path')
    parser.add_argument('--time_steps', type = int, default = 24)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument('--device', type = int, default = 0)

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    opt = parse_opt()
    data_dir = opt.data_dir
    location_dir = opt.location_dir
    save_path = opt.save_path
    time_steps = opt.time_steps
    epochs = opt.epochs
    batch_size = opt.batch_size
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else 'cpu')
    

    input_size = 28
    hidden_size = 28
    output_size = 1

    AQ_datasets = {x: Vanilla_LSTMDataset(os.path.join(data_dir, x), location_dir, time_steps)
                    for x in ['train', 'val']}
    AQ_dataloaders = {x: DataLoader(AQ_datasets[x], batch_size=batch_size, shuffle=True)
                    for x in ['train', 'val']}
    AQ_datasetsizes = {x: len(AQ_datasets[x]) for x in ['train', 'val']}
    
    lstm = Vanilla_LSTM(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(lstm.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_vanilla(lstm, AQ_dataloaders, AQ_datasetsizes,
                criterion, optimizer, exp_lr_scheduler,
                device, epochs, save_path)