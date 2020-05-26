import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args as parser_args

LearnedBatchNorm = nn.BatchNorm2d

def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x)*activation(torch.abs(x)-f(s))

def initialize_sInit():

    if parser_args.sInit_type == "constant":
        return parser_args.sInit_value*torch.ones([1, 1])

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class STRBatchNorm(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu
        if parser_args.sparse_function == 'sigmoid':
            self.f = torch.sigmoid
            self.sparseThreshold = nn.Parameter(initialize_sInit())
        else:
            self.sparseThreshold = nn.Parameter(initialize_sInit())

    def forward(self, x):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
        x = F.batch_norm(
            x, self.running_mean, self.running_var, sparseWeight, sparseWeight*self.bias, self.training, self.momentum, self.eps
        )
        return x

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), f(self.sparseThreshold).item()
