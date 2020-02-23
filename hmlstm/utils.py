from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from torch import nn
from torch.nn import Parameter


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, th: float) -> torch.Tensor:
        x[x >= th] = 1
        x[x < th] = 0

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dth = None  # indeterminable
        dx = grad_output  # identity/pass through gradient

        return dx, dth


class Round1(nn.Module):
    def __init__(self, th: float = 0.5):
        super(Round1, self).__init__()

        self.th = Parameter(torch.scalar_tensor(th), requires_grad=False)
        self.round = _Round.apply

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.round(x, self.th)


class HardSigm1(nn.Module):
    def __init__(self, a: int = 1):
        super(HardSigm1, self).__init__()

        self.a = Parameter(torch.scalar_tensor(a), requires_grad=False)

        # avoids "to" device call
        self.one = Parameter(torch.scalar_tensor(1), requires_grad=False)
        self.zero = Parameter(torch.scalar_tensor(0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min((self.a * x + 1) / 2, self.one), self.zero)


class HardSigm2(nn.Module):
    def __init__(self, a: int = 1):
        super(HardSigm2, self).__init__()

        self.a = Parameter(torch.scalar_tensor(a), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temp = torch.div(torch.add(torch.mul(x, self.a), 1), 2.0)
        output = torch.clamp(temp, min=0, max=1)
        return output


class SlopeScheduler():
    def __init__(self, params, factor: int = 25, max: int = 5) -> None:
        super(SlopeScheduler, self).__init__()

        self.params = params

        # a = min(5, a + factor * epoch)
        self.update = 1 / factor
        self.max = max
        self.name = "hardsigm.a"

    def step(self) -> None:
        for name, p in self.params:
            if not p.requires_grad and self.name in name:
                v = p.data + self.update
                p.data = v if v <= self.max else self.max


def plot_z(z: np.ndarray, data: List) -> None:
    L, S = z.shape

    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [0, 0.01, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(z, cmap=cmap, norm=norm)

    plt.xticks(range(S), data)
    plt.yticks(range(L), ["z" + str(l) for l in reversed(range(L))])

    plt.title(f"layers {L}, seq length {S}, \n boundary state (white ... z=1 / black ...z=0)")
    plt.rc('font', size=8)

    plt.show()


def plot_z(z: torch.Tensor, data: List = None) -> None:
    z = z.detach().numpy()

    for n in range(z.shape[0]):
        _plot_z(z[n].T, data)


def plot_h(h: torch.Tensor, layer_sizes: List[int], data: List = None) -> None:
    h = h.detach()

    h_layers = torch.split_with_sizes(h, layer_sizes, dim=2)
    h = [torch.norm(hl, dim=2) for hl in h_layers]
    h = torch.stack(h, dim=2)
    h = h.numpy()
    h = np.flip(h, axis=2)

    for n in range(h.shape[0]):
        _plot_h(h[n].T, data)


def plot_zh(z: torch.Tensor, h: torch.Tensor, layer_sizes: List[int], data: List = None) -> None:
    z = z.detach().numpy()

    h = h.detach()
    h_layers = torch.split_with_sizes(h, layer_sizes, dim=2)
    h = [torch.norm(hl, dim=2) for hl in h_layers]
    h = torch.stack(h, dim=2)
    h = h.numpy()
    h = np.flip(h, axis=2)

    _, S, L = z.shape
    for n in range(h.shape[0]):
        zh = np.dstack((z[n], h[n])).reshape((S, 2 * L))
        _plot_zh(zh.T, data)


def _plot_z(z: np.ndarray, data: List = None) -> None:
    L, S = z.shape

    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [0, 0.01, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # plt.rc('font', size=6)
    w, h = plt.figaspect(L / S)

    fig, ax = plt.subplots(figsize=(w, h))

    ax.imshow(z, cmap=cmap, norm=norm)

    xticks = data if data is not None else range(S)
    plt.xticks(range(S), xticks)
    plt.yticks(range(L), ["z" + str(l) for l in reversed(range(L))])

    plt.title(f"layers {L}, seq length {S}, \n boundary state (white ... z=1 / black ...z=0)")
    plt.show()


def _plot_h(h: np.ndarray, data: List = None) -> None:
    L, S = h.shape

    # plt.rc('font', size=6)
    width, height = plt.figaspect(L / S)

    fig, ax = plt.subplots(figsize=(width, height))
    ax.imshow(h)

    xticks = data if data is not None else range(S)
    plt.xticks(range(S), xticks)
    plt.yticks(range(L), ["h" + str(l) for l in reversed(range(L))])

    plt.title(f"hidden state norm (layers {L}, seq length {S})")

    plt.show()


def _plot_zh(zh: np.ndarray, data: List = None) -> None:
    L, S = zh.shape

    # plt.rc('font', size=6)
    w, h = plt.figaspect(L / S)

    fig, ax = plt.subplots(figsize=(w, h))
    ax.imshow(zh)

    xticks = data if data is not None else range(S)
    plt.xticks(range(S), xticks)
    yticks = ["h" + str(int(l / 2)) if l % 2 == 0 else "z" + str(int(l / 2)) for l in reversed(range(L))]
    plt.yticks(range(L), yticks)

    plt.title(f"hidden state norm and boundary state(z) (layers {int(L / 2)}, seq length {S})")
    plt.show()
