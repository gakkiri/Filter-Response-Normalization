import torch
import torch.nn as nn

parameter = nn.parameter.Parameter

class FRN(nn.Module):
    '''
    Filter Response Normalization Layer
    arxiv: https://arxiv.org/pdf/1911.09737.pdf
    '''

    def __init__(self, num_features, eps=1e-6, learnable=True, tlu=True):
        super(FRN, self).__init__()
        self.tlu = tlu

        self.beta = parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)
        self.gamma = parameter(torch.ones(1, num_features,1 ,1), requires_grad=True)

        if learnable:
            self.eps = parameter(torch.Tensor(1), requires_grad=True)
            nn.init.constant_(self.eps, eps)
        else:
            self.eps = torch.Tensor([eps])

        if tlu:
            self.tau = parameter(torch.zeros(1, num_features, 1, 1), requires_grad=True)

    def forward(self, x):

        nu2 = torch.mean(x.pow(2), (2, 3), keepdims=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        x = self.gamma * x + self.beta

        if self.tlu:
            x = torch.max(x, self.tau)

        return x


if __name__ == '__main__':
    frn = FRN(256)
    x = torch.randn(2, 256, 3, 3)
    print(x)
    x = frn(x)
    print(x)
