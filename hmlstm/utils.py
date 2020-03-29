from collections import defaultdict
from typing import List, Callable, Dict, Tuple

import re
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn
from torch.nn import Parameter

# TODO does not support jit for custom autograd functions https://discuss.pytorch.org/t/does-torch-jit-script-support-custom-operators/65759
from torch.utils.data import Dataset, DataLoader


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, th: float) -> torch.Tensor:
        # x[x >= th] = 1
        # x[x < th] = 0

        # x = torch.where(
        #     x >= th,
        #     torch.scalar_tensor(1, device=x.device),
        #     torch.scalar_tensor(0, device=x.device)
        # )

        # fastest version
        x = (x > th).float()

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
    """fastest version"""

    def __init__(self, a: int = 1):
        super(HardSigm2, self).__init__()

        self.a = Parameter(torch.scalar_tensor(a), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = 0.5 * (self.a * x + 1)

        return torch.clamp(t, min=0, max=1)


class SlopeScheduler():
    _name = "hardsigm.a"

    def __init__(self, params, factor: int = 25, max: int = 5) -> None:
        super(SlopeScheduler, self).__init__()

        self.params = params

        # a = min(5, a + factor * epoch)
        self.update = 1 / factor
        self.max = max

    def step(self) -> None:
        for name, p in self.params.items():
            # if not p.requires_grad and self._name in name:
            if self._name in name:
                v = p.data + self.update
                p.data = v if v <= self.max else torch.scalar_tensor(self.max)

    def set_slopes(self, slope: float):
        for name, p in self.params.items():
            if self._name in name:
                p = torch.scalar_tensor(slope)

    def get_slopes(self) -> List[float]:
        slopes = []
        for name, p in self.params.items():
            if self._name in name:
                slopes.append(float(p.data))

        return slopes

    def get_slope(self, idx: int = 0) -> float:
        return self.get_slopes()[idx]


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


def plot_z(z: torch.Tensor, data: List = None, yticks: List = None) -> None:
    z = z.detach().cpu().numpy()

    data = ["\\n" if v == "\n" else v for v in data]
    data = ["\\t" if v == "\t" else v for v in data]

    for n in range(z.shape[0]):
        _plot_z(z[n].T, data, yticks)


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


def _plot_z(z: np.ndarray, data: List = None, yticks_labels: List = None) -> None:
    L, S = z.shape

    xticks_labels = data if data is not None else range(S)
    yticks_labels = yticks_labels[::-1] if yticks_labels is not None else ["z" + str(l) for l in reversed(range(L))]

    if len(yticks_labels) < L:
        for l in reversed(range(L - len(yticks_labels))):
            yticks_labels.append("z" + str(l))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds = [0, 0.01, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # plt.rc('font', size=6)
    w, h = plt.figaspect(L / S) * 1.1

    fig, ax = plt.subplots(figsize=(w, h))

    ax.imshow(z, cmap=cmap, norm=norm)

    ax.set_xticks(range(S))
    ax.set_yticks(range(L))

    ax.set_xticklabels(xticks_labels)
    ax.set_yticklabels(yticks_labels)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(S + 1) - .5, minor=True)
    ax.set_yticks(np.arange(L + 1) - .5, minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="x", which="both", length=0)

    # ax.grid(which="minor", color="b", linestyle='-', linewidth=3)
    # ax.tick_params(axis="x", which="both", length=0)

    # plt.xticks(range(S), xticks_labels)
    # plt.yticks(range(L), yticks_labels)

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


def _get_boundaries(string: str, regex: str = " ", end: bool = True, offset: int = 1) -> np.ndarray:
    if "random" in regex:
        return np.random.binomial(1, p=float(regex[7:]), size=len(string))

    ret = np.zeros(len(string))

    for m in re.finditer(r"(" + regex + ")", string):
        if end:
            ret[m.end() - offset] = 1
        else:
            ret[m.start()] = 1

    return ret


def get_boundaries(metrics: Dict, text: str, seq_len: int, device: torch.device = "cpu") -> Dict:
    for name, regex in metrics.items():
        b = _get_boundaries(text, regex)
        f = int(len(b) / seq_len)
        b = b[:f * seq_len]
        b = np.vstack(np.split(b, f))

        metrics[name] = torch.from_numpy(b).float().to(device)

    return metrics


def cross_entropy(p: torch.Tensor, q: torch.Tensor, eps=0.00001) -> float:
    p[p == 0] = eps
    q[q == 0] = eps

    return -torch.mean(p * torch.log(q)).item()


def statistical_metrics(y: torch.Tensor, y_hat: torch.Tensor):
    correct = torch.eq(y, y_hat).sum().item()

    confusion_vector = y_hat / y
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    acc = correct / y.numel()

    f1_score = (2 * tp) / (2 * tp + fp + fn)

    return f1_score, acc


def evaluate_z(model: nn.Module, dataset: Dataset, metrics: Dict, batch_size,
               device: torch.device = "cpu", build_metrics: bool = True) -> Tuple[Dict, Dict]:
    """function evaluates z on the whole dataset based on given metrics"""
    if build_metrics:
        metrics = get_boundaries(metrics, dataset.text, dataset.seq_length, device)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device, drop_last=True)
    results = defaultdict(lambda: torch.zeros(model.num_layers, len(dl), device=device))

    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (inputs, _, idx) in enumerate(dl):
            #print(f"{i}/{len(dl)}")
            inputs = inputs.to(device)

            _, (_, z) = model(inputs)

            for name, boundary in metrics.items():
                for l in range(model.num_layers):
                    # _print(inputs[idx[20]], z[idx[20], :, l], boundary[idx[20]], name + "_" + str(l))
                    results[name + "_f1_score"][l, i], results[name + "_accuracy"][l, i] = statistical_metrics(
                        boundary[idx], z[:, :, l])
                    results[name + "_cross_entropy"][l, i] = cross_entropy(boundary[idx], z[:, :, l])

    return {name: torch.mean(values, dim=1) for name, values in results.items()}, metrics


def _print(x, z, b, name, dataset, l: int = 20):
    to_str = lambda arr: re.sub("[\n|\[|\] ]", "", np.array2string(arr.astype(np.int)))
    text = dataset.decode(x)
    z = z.cpu().numpy()
    b = b.cpu().numpy()

    print("text : ".rjust(l) + text.replace("\n", "|"))
    print("z_bound : ".rjust(l) + to_str(z))
    print((name + " : ").rjust(l) + to_str(b))


def get_z(input: torch.Tensor, model: nn.Module, dataset: Dataset, device: torch.device = "cpu") -> Tuple[
    torch.Tensor, str]:
    model.to(device)
    model.eval()
    with torch.no_grad():
        input = input.to(device).unsqueeze(dim=0)

        _, (_, z) = model(input)

    return z, dataset.decode(input.squeeze())


def stack_z(metrics: Dict, i: int, seq_len: int, z: torch.Tensor = None) -> torch.Tensor:
    for name, boundaries in metrics.items():
        b = boundaries[i].reshape(1, seq_len, 1)
        z = b if z is None else torch.cat((b, z), dim=2)

    return z
