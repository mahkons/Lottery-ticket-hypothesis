import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse

from load_data import load_train_data
from networks.SimpleNet import SimpleNet
from networks.VGG19 import vgg19
from PruningWrapper import PruningWrapper

from networks.SGD import SGD

def train(dataloader, epochs, model, optimizer, criterion, device):

    plot_data = list()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(enumerate(dataloader, 0))
        for i, data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            #  model.zero_unmasked_grad()
            with torch.no_grad():
                optimizer.step()

            plot_data.append(loss.item())
            running_loss += loss.item()
            if i % 500 == 499:
                pbar.write('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0

    model.save_model("generated/" + model.get_path())
    torch.cuda.empty_cache()
    return plot_data


def iterative_pruning(model, iters, device):
    trainloader = load_train_data(batch_size=4)
    criterion = nn.CrossEntropyLoss()

    for iter in tqdm(range(iters)):
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train(trainloader, 1, model, optimizer, criterion, device)
        model.prune_net(20)
        model.reinit_net()


def train_no_pruning(model, epochs, device):
    trainloader = load_train_data(batch_size=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    plot_data = train(trainloader, epochs, model, optimizer, criterion, device)
    return plot_data


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--pruning-iters', type=int, default=0, required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)

    #  wrapper = PruningWrapper(SimpleNet(), device)
    wrapper = SimpleNet().to(device)
    #  wrapper = PruningWrapper.load_model(SimpleNet, "generated/" + SimpleNet.__name__, device)

    #  iterative_pruning(wrapper, args.pruning_iters, device)
    plot_data = train_no_pruning(wrapper, 1, device)
    torch.save(plot_data, "plots/%.3f" % ((0.8 ** args.pruning_iters)*100))

