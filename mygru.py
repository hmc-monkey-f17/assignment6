import math
import torch
import torch.nn as nn

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.w_ih = nn.Parameter(torch.Tensor(num_layers, hidden_size * 3, input_size))
        self.b_ih = nn.Parameter(torch.Tensor(num_layers, 1, hidden_size * 3))
        self.w_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size * 3, hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(num_layers, 1, hidden_size * 3))
        self.init_hidden()
        
    def init_hidden(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        # This initialization helps the speed of learning.
        # Contrast with doing a normal distribution which learns much
        # more slowly.
        nn.init.uniform(self.weight_ih_l0, -stdv, stdv)
        nn.init.uniform(self.bias_ih_l0, -stdv, stdv)
        nn.init.uniform(self.weight_hh_l0, -stdv, stdv)
        nn.init.uniform(self.bias_hh_l0, -stdv, stdv)

    def forward(self, inp, h_0):
        seq_len = inp.shape[0]
        
        # iterate seq_len times, calculating h_i at each timestep
        
        h_i = h_0
        output = torch.Tensor(seq_len, self.hidden_size)

        for i in range(seq_len):  
            Z_X = torch.bmm(inp[i], self.w_ih) + self.b_ih
            Z_H = torch.bmm(h_i,    self.w_hh) + self.b_hh
            r_i = torch.sigmoid(Z_X.narrow(0, 0, self.hidden_size) 
                              + Z_H.narrow(0, 0, self.hidden_size))
            z_i = torch.sigmoid(Z_X.narrow(0, self.hidden_size, self.hidden_size) 
                              + Z_H.narrow(0, self.hidden_size, self.hidden_size))
            n_i = torch.tanh(Z_X.narrow(0, 2 * self.hidden_size, self.hidden_size)
                     + r_i * Z_H.narrow(0, 2 * self.hidden_size, self.hidden_size))
            h_i = (1 - z_i) * n_i + z_i * h_i
            output[i] = h_i[self.num_layers - 1][0]

        return output, h_i
