import math
import torch
import torch.nn as nn
from torch.autograd import Variable

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1,
                dropout=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.w_ih_l0 = nn.Parameter(torch.Tensor(hidden_size * 3, input_size)).cuda()
        self.b_ih_l0 = nn.Parameter(torch.Tensor(hidden_size * 3)).cuda()
        if num_layers > 1:
            self.w_ih = nn.Parameter(torch.Tensor(num_layers-1, hidden_size * 3, hidden_size)).cuda()
            self.b_ih = nn.Parameter(torch.Tensor(num_layers-1, hidden_size * 3)).cuda()
        self.w_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size * 3, hidden_size)).cuda()
        self.b_hh = nn.Parameter(torch.Tensor(num_layers, hidden_size * 3)).cuda()
        self.init_hidden()
        
    def init_hidden(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)

        # This initialization helps the speed of learning.
        # Contrast with doing a normal distribution which learns much
        # more slowly.
        nn.init.uniform(self.w_ih_l0, -stdv, stdv)
        nn.init.uniform(self.b_ih_l0, -stdv, stdv)
        if self.num_layers > 1:
            nn.init.uniform(self.w_ih, -stdv, stdv)
            nn.init.uniform(self.b_ih, -stdv, stdv)
        nn.init.uniform(self.w_hh, -stdv, stdv)
        nn.init.uniform(self.b_hh, -stdv, stdv)

    def forward(self, inp, h_0):
        seq_len = inp.shape[0]
        bs = inp.shape[1]
        # iterate seq_len times, calculating h_i at each timestep
       
        if torch.cuda.is_available():
            inp = inp.cuda()
            h_0 = h_0.cuda()

        h = None

        if seq_len:
            Z_X = torch.mm(inp[0], self.w_ih_l0.t()) + self.b_ih_l0.unsqueeze(0)
            Z_H = torch.mm(h_0[0], self.w_hh[0].t()) + self.b_hh[0].unsqueeze(0)
            r_i = torch.sigmoid(Z_X.narrow(1, 0, self.hidden_size)
                              + Z_H.narrow(1, 0, self.hidden_size))
            z_i = torch.sigmoid(Z_X.narrow(1, self.hidden_size, self.hidden_size)
                              + Z_H.narrow(1, self.hidden_size, self.hidden_size))
            n_i = torch.tanh(Z_X.narrow(1, 2 * self.hidden_size, self.hidden_size)
                     + r_i * Z_H.narrow(1, 2 * self.hidden_size, self.hidden_size))
            h_row = ((1 - z_i) * n_i + z_i * h_0[0]).unsqueeze(0)

            for j in range(1,self.num_layers):
                Z_X = torch.mm(h[0][j-1], self.w_ih[j-1].t()) + self.b_ih[j-1].unsqueeze(0)
                Z_H = torch.mm(h_0[j],    self.w_hh[j].t())   + self.b_hh[j].unsqueeze(0)
                r_i = torch.sigmoid(Z_X.narrow(1, 0, self.hidden_size)
                                  + Z_H.narrow(1, 0, self.hidden_size))
                z_i = torch.sigmoid(Z_X.narrow(1, self.hidden_size, self.hidden_size)
                                  + Z_H.narrow(1, self.hidden_size, self.hidden_size))
                n_i = torch.tanh(Z_X.narrow(1, 2 * self.hidden_size, self.hidden_size)
                         + r_i * Z_H.narrow(1, 2 * self.hidden_size, self.hidden_size))
                h_row = torch.cat((h_row,((1 - z_i) * n_i + z_i * h_0[j]).unsqueeze(0)),0)
            
            h = h_row.unsqueeze(0)
        else:
            return None, None


        for i in range(1,seq_len):
            Z_X = torch.mm(inp[i],    self.w_ih_l0.t()) + self.b_ih_l0.unsqueeze(0)
            Z_H = torch.mm(h[i-1][0], self.w_hh[0].t()) + self.b_hh[0].unsqueeze(0)
            r_i = torch.sigmoid(Z_X.narrow(1, 0, self.hidden_size)
                              + Z_H.narrow(1, 0, self.hidden_size))
            z_i = torch.sigmoid(Z_X.narrow(1, self.hidden_size, self.hidden_size)
                              + Z_H.narrow(1, self.hidden_size, self.hidden_size))
            n_i = torch.tanh(Z_X.narrow(1, 2 * self.hidden_size, self.hidden_size)
                     + r_i * Z_H.narrow(1, 2 * self.hidden_size, self.hidden_size))
            h_row = ((1 - z_i) * n_i + z_i * h[i-1][0]).unsqueeze(0)

            for j in range(1,self.num_layers):
                Z_X = torch.mm(h[i][j-1], self.w_ih[j-1].t()) + self.b_ih[j-1].unsqueeze(0)
                Z_H = torch.mm(h[i-1][j], self.w_hh[j].t())   + self.b_hh[j].unsqueeze(0)
                r_i = torch.sigmoid(Z_X.narrow(1, 0, self.hidden_size)
                                  + Z_H.narrow(1, 0, self.hidden_size))
                z_i = torch.sigmoid(Z_X.narrow(1, self.hidden_size, self.hidden_size)
                                  + Z_H.narrow(1, self.hidden_size, self.hidden_size))
                n_i = torch.tanh(Z_X.narrow(1, 2 * self.hidden_size, self.hidden_size)
                         + r_i * Z_H.narrow(1, 2 * self.hidden_size, self.hidden_size))
                h_row = torch.cat((h_row,((1 - z_i) * n_i + z_i * h[i-1][j]).unsqueeze(0)),0)
            h = torch.cat((h,h_row.unsqueeze(0)),0)

        output = h[:,self.num_layers-1].clone()
        h_t = h[seq_len-1, :].clone()

        return output, h_t
