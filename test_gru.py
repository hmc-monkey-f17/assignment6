import numpy as np
import torch
from torch.autograd import Variable
import mygru
import unittest


# Suggested to run these tests with nosetests
# If you do that, print statements will be ignored in passing tests (only print
# in failing tests).
#
# You can also run python test_gru.py, but that'll print everything for all tests
# (passing or failing).

# modify this to GRU_CLASS = MyGRU to test *your* class
#GRU_CLASS = mygru.MyGRU
GRU_CLASS = torch.nn.GRU

class TestGRU(unittest.TestCase):
    def test_simplest_input_has_right_shapes(self):
        input_size = 1
        hidden_size = 1
        gru = GRU_CLASS(input_size, hidden_size)
        
        num_layers = 1
        num_directions = 1
        bs = 1  # batch_size
        seq_len = 1
        inp = Variable(torch.Tensor(np.array([[[5]]]))) # one input that is 5
        h_0 = Variable(torch.Tensor(np.zeros((num_layers*num_directions, bs, hidden_size))))
        print("inp", inp, inp.shape)
        print("h_0", h_0, h_0.shape)

        outp, h = gru(inp, h_0)
        print("outp", outp, outp.shape)
        print("h", h, h.shape)
        self.assertEqual((seq_len, bs, num_directions * hidden_size), outp.shape)
        self.assertEqual((num_layers * num_directions, bs, hidden_size), h.shape)

    def test_larger_input_has_right_shapes(self):
        input_size = 3
        hidden_size = 4
        gru = GRU_CLASS(input_size, hidden_size)
        
        num_layers = 1
        num_directions = 1
        bs = 2  # batch_size
        seq_len = 5
        inp = Variable(torch.Tensor(np.zeros((seq_len, bs, input_size))))
        print("inp", inp, inp.shape)
        h_0 = Variable(torch.Tensor(np.zeros((num_layers*num_directions, bs, hidden_size))))
        print("h_0", h_0, h_0.shape)
        outp, h = gru(inp, h_0)
        print("outp", outp, outp.shape)
        print("h", h, h.shape)
        self.assertEqual((seq_len, bs, num_directions * hidden_size), outp.shape)
        self.assertEqual((num_layers * num_directions, bs, hidden_size), h.shape)


    def test_h_is_unchanged(self):
        input_size = 1
        hidden_size = 1
        gru = GRU_CLASS(input_size, hidden_size)
        bias_hh_0 = gru.bias_hh_l0 # Hidden-hidden biases for first layer
        bias_hh_0.data[1] = 10000000 # set b_hz to high number so that z will be close to 1
        
        num_layers = 1
        num_directions = 1
        bs = 1  # batch_size
        seq_len = 1
        inp = Variable(torch.Tensor(np.zeros((seq_len, bs, input_size))))
        inp[0, 0, 0] = 2
        h_0 = Variable(torch.Tensor(np.zeros((num_layers*num_directions, bs, hidden_size))))
        h_0[0, 0, 0] = 53
        outp, h = gru(inp, h_0)
        self.assertEqual((seq_len, bs, num_directions * hidden_size), outp.shape)
        self.assertEqual((num_layers * num_directions, bs, hidden_size), h.shape)
        self.assertEqual(53, h.data[0, 0, 0])

    def test_multiple_hidden_units_multiple_input_size(self):
        torch.manual_seed(829)  # make this test 100% reproducible

        input_size = 2
        hidden_size = 3
        gru = GRU_CLASS(input_size, hidden_size)

        # Extract weights and biases from the GRU_CLASS
        bias_ih_0 = gru.bias_ih_l0
        bias_hh_0 = gru.bias_hh_l0 
        b_ir = bias_ih_0[:hidden_size].unsqueeze(0).transpose(0, 1)
        b_iz = bias_ih_0[hidden_size:2*hidden_size].unsqueeze(0).transpose(0, 1)
        b_in = bias_ih_0[2*hidden_size:].unsqueeze(0).transpose(0, 1)
        b_hr = bias_hh_0[:hidden_size].unsqueeze(0).transpose(0, 1)
        b_hz = bias_hh_0[hidden_size:2*hidden_size].unsqueeze(0).transpose(0, 1)
        b_hn = bias_hh_0[2*hidden_size:].unsqueeze(0).transpose(0, 1)
        print("bias_ih_0", bias_ih_0, bias_ih_0.shape)
        print("b_ir", b_ir, b_ir.shape)
        print("b_iz", b_iz, b_iz.shape)
        print("b_in", b_in, b_in.shape)
        print("bias_hh_0", bias_hh_0, bias_hh_0.shape)
        print("b_hr", b_hr, b_hr.shape)
        print("b_hz", b_hz, b_hz.shape)
        print("b_hn", b_hn, b_hn.shape)

        weight_ih_0 = gru.weight_ih_l0
        weight_hh_0 = gru.weight_hh_l0
        w_ir = weight_ih_0[:hidden_size]
        w_iz = weight_ih_0[hidden_size:2*hidden_size]
        w_in = weight_ih_0[2*hidden_size:]
        print("weight_ih_0", weight_ih_0, weight_ih_0.shape)
        print("w_ir", w_ir, w_ir.shape)
        print("w_iz", w_iz, w_iz.shape)
        print("w_in", w_in, w_in.shape)

        w_hr = weight_hh_0[:hidden_size]
        w_hz = weight_hh_0[hidden_size:2*hidden_size]
        w_hn = weight_hh_0[2*hidden_size:]
        print("weight_hh_0", weight_hh_0, weight_hh_0.shape)
        print("w_hr", w_hr, w_hr.shape)
        print("w_hz", w_hz, w_hz.shape)
        print("w_hn", w_hn, w_hn.shape)

        num_layers = 1
        num_directions = 1
        bs = 1  # batch_size
        seq_len = 1
        inp = Variable(torch.Tensor(np.zeros((seq_len, bs, input_size))))
        inp[0, 0, 0] = 2
        h_0 = Variable(torch.Tensor(np.zeros((num_layers*num_directions, bs, hidden_size))))
        h_0[0, 0, 0] = 1
        h_0[0, 0, 1] = 2
        h_0[0, 0, 2] = 3
        print("h_0", h_0, h_0.shape)
        print("inp", inp, inp.shape)

        inp_0 = inp[0,:].transpose(0, 1)  # inp holds all inputs.  We want just the first one tranposed
        h_forward_0 = h_0[0,:].transpose(0, 1) # h_0 could hold both forward and backward. We want forward
        print("inp_0", inp_0, inp_0.shape)
        print("h_forward_0", h_forward_0, h_forward_0.shape)
        r = torch.sigmoid(torch.mm(w_ir, inp_0) + b_ir + torch.mm(w_hr, h_forward_0) + b_hr)
        print("r", r, r.shape)
        z = torch.sigmoid(torch.mm(w_iz, inp_0) + b_iz + torch.mm(w_hz, h_forward_0) + b_hz)
        print("z", z, z.shape)
        n = torch.tanh(torch.mm(w_in, inp_0) + b_in +  r*(torch.mm(w_hn, h_forward_0) + b_hn))
        print("n", n, n.shape)
        h_t = (1-z)*n + z*h_0[0].transpose(0, 1)
        print("h_t", h_t, h_t.shape)

        outp, h = gru(inp, h_0)
        
        print("outp", outp, outp.shape)
        print("h", h, h.shape)

        self.assertAlmostEqual(h_t.data[0, 0], h.data[0, 0, 0], places=5)  # We use assertAlmostEqual for floats
        self.assertAlmostEqual(h_t.data[1, 0], h.data[0, 0, 1], places=5)  # We use assertAlmostEqual for floats
        self.assertAlmostEqual(h_t.data[2, 0], h.data[0, 0, 2], places=5)  # We use assertAlmostEqual for floats


    # Still needed: test for seq_len > 1, and batch_size > 1

if __name__ == '__main__':
    unittest.main()

