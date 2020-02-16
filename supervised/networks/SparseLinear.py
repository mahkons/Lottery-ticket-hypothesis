import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import math


# Still does not work correctly with any optimizer
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.empty(out_features, in_features)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight = Parameter(self.weight.to_sparse())

        self.bias = torch.Tensor(out_features)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        self.bias = Parameter(self.bias)

    def forward(self, input):
        return torch.sparse.mm(self.weight, input.T).T + self.bias


if __name__ == "__main__":
    x = torch.tensor([[1, 2]], dtype=torch.float)
    a = SparseLinear(2, 3)
    #  b = Parameter(torch.tensor([[1, 2], [-1, -2]], dtype=torch.float).to_sparse())

    #  optimizer = optim.SparseAdam(a.parameters(), lr=1e-3)
    #  optimizer = optim.Adam(a.parameters())
    from SGD import SGD

    optimizer = SGD(a.parameters(), lr=1e-3)
    #  loss = a(x).sum()
    loss = a(x).sum()
    loss.backward()
    optimizer.step()

    print(a.weight)
    with torch.no_grad():
        a.weight.add_(a.weight.grad)
    print(a.weight)
