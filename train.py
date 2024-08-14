import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataset import Dataset
from tqdm import tqdm

import argparse
import os
import csv

config = {
    'train_dataset': Dataset('./train.txt'),
    'val_dataset': Dataset('./val.txt'),
    'lr': 5e-4,
    'epochs': 2,
    'batch_size': 128
}

train_record = [['epoch', 'batch_idx', 'loss']]
val_record = []


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    epochs = config['epochs']

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.BCELoss()

    train_dataset = config['train_dataset']
    val_dataset = config['val_dataset']
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    if not os.path.exists('models'):
        os.mkdir('models')

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        val(model, device, val_loader, criterion)
        torch.save(model.state_dict(), 'models/epoch-{}.pth'.format(epoch))

    with open('train_record.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(train_record)
    with open('val_record.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(val_record)


def train(model, device, train_loader, optimizer, criterion, epoch):
    # TODO: lr decay
    model.train()
    train_loader = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loader.set_description(
            'Train Epoch: {}/{}'.format(epoch, config['epochs'])
        )
        train_loader.set_postfix(loss=loss.item())
        train_record.append([epoch, batch_idx + 1, loss.item()])


def val(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device).float().unsqueeze(1)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
    val_record.append([val_loss])
    return val_loss
    # TODO: add early stopping


if __name__ == '__main__':
    main()
