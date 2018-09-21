#!/usr/bin/env python3
"""Derived largely from https://stackoverflow.com/questions/50168224/does-a-clean-and-extendable-lstm-implementation-exists-in-pytorch.
"""

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

        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        # x is of dimensionality (batch, input_size)
        if hidden is None:
            hidden = self._init_hidden(x)
        h, c = hidden

        preact = self.i2h(x) + self.h2h(h)

        gates = preact[:, :3 * self.hidden_size].sigmoid()
        g_t = preact[:, 3 * self.hidden_size:].tanh()
        i_t = gates[:, :self.hidden_size]
        f_t = gates[:, self.hidden_size:2 * self.hidden_size]
        o_t = gates[:, -self.hidden_size:]

        c_t = th.mul(c, f_t) + th.mul(i_t, g_t)

        h_t = th.mul(o_t, c_t.tanh())

        return h_t, (h_t, c_t)

    def _init_hidden(self, input_):
        h = th.zeros(input_.size(0), self.hidden_size)
        c = th.zeros(input_.size(0), self.hidden_size)
        return h, c


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTM, self).__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.linear = nn.Linear(hidden_size, 1)

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

    optimizer = optim.SGD(lstm.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in tqdm(range(args.epochs)):
        Y_hat = lstm(X)
        loss = criterion(Y_hat, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 99:
            print(loss.item())

    X = th.from_numpy(np.random.binomial(1, p=p, size=(50, 1, 1))).float()
    print(X.detach().numpy().flatten())
    print(np.round(lstm(X).detach().numpy().flatten(), 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
