import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse

from load_data import load_train_data
from networks.SimpleNet import SimpleNet
from PruningWrapper import PruningWrapper

def train(dataloader, epochs, model, optimizer, criterion, device):

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(dataloader, 0))
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            model.zero_unmasked_grad()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                pbar.write('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    model.save_model("generated/" + model.get_path())
    torch.cuda.empty_cache()


def iterative_prunning(model, device):
    trainloader = load_train_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for iter in range(5):
        train(trainloader, 10, model, optimizer, criterion, device)
        if (iter != 4):
            model.prune_net(10)
            model.reinit_net()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)

    wrapper = PruningWrapper(SimpleNet(), device)
    iterative_prunning(wrapper, device)
