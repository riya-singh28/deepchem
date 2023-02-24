
import os
import unittest
import deepchem as dc
import numpy as np
import pytest
try:
    import tensorflow as tf
    from tensorflow.python.eager import context
    has_tensorflow = True
except:
    has_tensorflow = False

from torch import nn as nn
from torch.nn import init as initializers
import torch


class EdgeNetwork(nn.Module):
    """ Submodule for Message Passing """

    def __init__(self,
                 n_pair_features=8,
                 n_hidden=100,
                 init='xavier_uniform_',
                 **kwargs):
        super(EdgeNetwork, self).__init__(**kwargs)
        self.n_pair_features = n_pair_features
        self.n_hidden = n_hidden
        self.init = init

        #def init(input_shape):
        #    return self.add_weight(name='kernel',
        #                           shape=(input_shape[0], input_shape[1]),
        #                           initializer=self.init,
        #                           trainable=True)
        init = getattr(initializers, self.init)
        n_pair_features = self.n_pair_features
        n_hidden = self.n_hidden
        self.W = init(torch.empty([n_pair_features, n_hidden * n_hidden]))
        self.b = torch.zeros((n_hidden * n_hidden,))
        self.built = True


    def __repr__(self) -> str:
        return (
        f'{self.__class__.__name__}(n_pair_features:{self.n_pair_features},n_hidden:{self.n_hidden},init:{self.init})'
    )

    def forward(self, inputs):
        pair_features, atom_features, atom_to_pair = inputs
        A = torch.add(torch.matmul(pair_features, self.W), self.b)
        A = torch.reshape(A, (-1, self.n_hidden, self.n_hidden))
        out = torch.randn(torch.gather(atom_features, atom_to_pair[:, 1]), 2)
        out = torch.squeeze(torch.matmul(A, out), axis=2)
        return torch.math.segment_sum(out, atom_to_pair[:, 0])
        


pair_features = torch.rand((868, 14), dtype=torch.float32)
atom_features = torch.rand((2945, 75), dtype=torch.float32)
atom_to_pair = torch.rand((868, 2), dtype=torch.int32)
inputs = [pair_features, atom_features, atom_to_pair]
#print(inputs)

#t = test_edge_network()
n_pair_features = 14
n_hidden = 75
init = 'xavier_uniform_'
layer = EdgeNetwork(n_pair_features, n_hidden, init)
t = layer(inputs)
print("riya")
print(t.shape)

#
#

#(86835, 14) (2945, 75) (86835, 2)

# %%
