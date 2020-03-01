import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse

from load_data import load_test_data
from networks.SimpleNet import SimpleNet
from networks.VGG19 import VGG19, vgg19

def test(net, criterion, device):
    testloader = load_test_data()

    correct, total = 0, 0
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    torch.cuda.empty_cache()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', required=False)
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)

    net = vgg19().load_model("generated/" + VGG19.__name__).to(device)
    criterion = nn.CrossEntropyLoss()
    test(net, criterion, device)
