import math
import signal
from typing import Callable

import torch
import numpy as np
from IPython import get_ipython
from torch import nn
from torch.distributions import Multinomial
from torch.utils.data import Dataset


# adapted from https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
class GracefulExit:
    exit_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.exit_now = True


def print_cuda_info():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"cuda device: {i} / "
                  f"name: {torch.cuda.get_device_name(i)} / "
                  f"cuda-capability: {torch.cuda.get_device_capability(i)} / "
                  f"memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)} GB")
    else:
        print("cuda is not available")


def predict(encoded_text: torch.Tensor, model: nn.Module, k: int = 1, device: torch.device = "cpu") -> torch.Tensor:
    model.eval()

    (out) = model(encoded_text.to(device))

    logits = out[0]

    # TODO why?
    logits = logits[:, -1]
    sample = Multinomial(k, logits=logits).sample()
    prediction = sample.argmax().reshape((encoded_text.shape[0],))

    return prediction


def sample_text(text: str, length: int, model: nn.Module, dataset: Dataset, k: int = 1, online=True,
                device: torch.device = "cpu") -> str:
    # print("we are sampling on", device, end="\n", flush=True)

    text = dataset.preprocess_text(text)
    T = len(text)

    encoded_text = dataset.encode(text).reshape((1, T, 1)).to(device)

    if online:
        print(text, end="")

    for i in range(length):
        prediction = predict(encoded_text, model, k)
        encoded_text = torch.cat((encoded_text, prediction.reshape(1, 1, 1)), dim=1)

        if online:
            print(dataset.decode(prediction), end="")

    return dataset.decode(encoded_text.squeeze())


# adapted from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class SimpleEarlyStopping(object):
    def __init__(self, patience: int = 10, delta: float = .0, verbose: bool = False) -> None:
        super().__init__()

        self.patience = patience
        self.delta = delta
        self.min_loss = float("inf")

    # TODO callback save checkpoint
    def __call__(self, loss: torch.Tensor, save_checkpoint: Callable = None) -> bool:
        if self._check(loss):
            self.min_loss = loss

            if save_checkpoint is not None:
                save_checkpoint()
        else:
            self.patience -= 1

        return self.patience < 0

    def _check(self, loss) -> bool:
        return loss + self.delta <= self.min_loss


class SMAEarlyStopping(SimpleEarlyStopping):
    def __init__(self, patience: int = 10, delta: float = .0, verbose: bool = False, n: int = 10) -> None:
        super(SMAEarlyStopping, self).__init__(patience, delta, verbose)

        self.losses = torch.zeros(n)
        self.i = 0

    def _check(self, loss: torch.Tensor) -> bool:
        self.losses[self.i] = loss
        self.i = self.i + 1 if self.i < len(self.losses) - 1 else 0

        return loss + self.delta <= torch.mean(self.losses)


class EMAEarlyStopping(SimpleEarlyStopping):
    def __init__(self, patience: int = 10, delta: float = .0, verbose: bool = False, n: int = 10) -> None:
        super(EMAEarlyStopping, self).__init__(patience, delta, verbose)

        self.losses = torch.zeros(n)
        self.i = 0

    def _check(self, loss: torch.Tensor) -> bool:
        self.losses[self.i] = loss
        self.i = self.i + 1 if self.i < len(self.losses) - 1 else 0

        # TODO implement
        #return loss + self.delta <= torch.mean(self.losses)
        raise NotImplementedError
