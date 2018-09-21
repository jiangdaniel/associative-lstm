#!/usr/bin/env python3

import math
import argparse
from tqdm import tqdm

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.i2h = nn.Linear(input_size, 7 * hidden_size, bias=bias)
        self.h2h = nn.Linear(2 * hidden_size, 7 * hidden_size, bias=bias)

        self.i2u = nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2u = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def bound(self, u):
        """Hard bound complex vector U to have values between 0 and 1"""
        u_real, u_imag = th.split(u, self.hidden_size, dim=1)
        d = th.clamp(
                th.sqrt(th.mul(u_real, u_real) + th.mul(u_imag, u_imag)),
                min=1)
        d = d.repeat(1, 2)
        return u / d

    def complex_mul(self, u, v):
        """Complex multiplication of vectors U and V"""
        u_real, u_imag = th.split(u, self.hidden_size, dim=1)
        v_real, v_imag = th.split(v, self.hidden_size, dim=1)
        return th.cat((
                th.mul(u_real, v_real) - th.mul(u_imag, v_imag),
                th.mul(u_real, v_imag) + th.mul(u_imag, v_real)),
                dim=1)

    def forward(self, x, hidden):
        # x is of dimensionality (batch, input_size)
        if hidden is None:
            hidden = self._init_hidden(x)
        h, c = hidden

        preact = self.i2h(x) + self.h2h(h)

        gates, keys = th.split(preact, (3 * self.hidden_size, 4 * self.hidden_size), dim=1)

        g_f, g_i, g_o = th.split(gates.sigmoid(), self.hidden_size, dim=1)
        g_f, g_i, g_o = g_f.repeat(1, 2), g_i.repeat(1, 2), g_o.repeat(1, 2)
        r_i, r_o = th.split(keys, 2 * self.hidden_size, dim=1)

        u = self.bound(self.i2u(x) + self.h2u(h))
        r_i = self.bound(r_i)
        r_o = self.bound(r_o)

        c_t = th.mul(g_f, c) + self.complex_mul(r_i, th.mul(g_i, u))
        h_t = th.mul(g_o, self.bound(self.complex_mul(r_o, c_t)))

        return h_t, (h_t, c_t)

    def _init_hidden(self, input_):
        h = th.zeros(input_.size(0), 2 * self.hidden_size)
        c = th.zeros(input_.size(0), 2 * self.hidden_size)
        return h, c


class LSTM(nn.Module):
    """https://stackoverflow.com/questions/50168224/does-a-clean-and-extendable-lstm-implementation-exists-in-pytorch"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.linear = nn.Linear(hidden_size*2, 1)

    def forward(self, input_, hidden=None):
        # input_ is of dimensionalty (time, batch, input_size)
        outputs = []
        for x in th.unbind(input_, dim=0):
            out, hidden = self.lstm_cell(x, hidden)
            out = self.linear(out).sigmoid()
            outputs.append(out)
        return th.stack(outputs, dim=0)


def main(args):
    """Test on task where it should output 1 whenever it sees three 1s in a row."""
    criterion = nn.BCELoss()
    N = 1000
    T = 100
    p = 0.75
    lstm = LSTM(1, 5)
    X = th.from_numpy(np.random.binomial(1, p=p, size=(T, N, 1))).float()
    Y = th.from_numpy(np.zeros((T, N, 1))).float()
    for n in range(N):
        count = 0
        for t in range(T):
            if X[t, n, 0] == 1:
                count += 1
            else:
                count = 0
            if count >= 3:
                Y[t, n, 0] = 1

    optimizer = optim.Adam(lstm.parameters(), lr=args.lr)
    for epoch in tqdm(range(args.epochs)):
        Y_hat = lstm(X)
        loss = criterion(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 9:
            print(loss.item())

    X = th.from_numpy(np.random.binomial(1, p=p, size=(50, 1, 1))).float()
    print(X.detach().numpy().flatten())
    print(np.round(lstm(X).detach().numpy().flatten(), 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
