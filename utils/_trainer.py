import signal
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch.distributions import Multinomial
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from hmlstm.utils import SlopeScheduler


class _Trainer(object):
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, loss: nn.Module = nn.CrossEntropyLoss(),
                 device: torch.device = "cpu", path_save: str = None, fn_save: str = None):

        # TODO check if this works
        self.exit = GracefulExit()
        self.path_save = path_save
        self.fn_save = fn_save

        # TODO maybe add optimizer as init parameter
        self.device = device
        self.model = model
        self.loss = loss
        self.optimizer = torch.optim.Adam(model.parameters())

        # self.slope_scheduler = SlopeScheduler(model.named_parameters())

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset = train_loader

        self.train_losses = {}
        self.val_losses = {}
        self.epoch = 0

    def get_validation_loss(self):
        val_losses = []

        self.model.eval()

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits, _, _ = self.model(x)

                # TODO how to concat validation loss for optim scheduler?
                val_loss = self.loss(logits.view(-1, logits.shape[2]), y.view(-1))

                val_losses.append(val_loss.data.mean())

        self.model.train()
        return sum(val_losses) / len(val_losses), val_loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def print_info(self):

        print("")
        print("we are training on", self.device, end="\n", flush=True)

        self.print_cuda_info()

    def train(self, epoch: int, lr: int = 0.01, log_interval: int = 50, validate: bool = False):
        N = len(self.train_loader.dataset)

        print("training samples per epoch  : %d" % (N))
        print("training on samples (total) : %d" % (N * epoch))

        self.to(self.device)
        self.set_lr(lr)
        self.model.train()

        self.train_losses = {}
        self.val_losses = {}

        # TODO use right tqdm if in jupyter notebook
        pbar = tqdm.tqdm(total=(N * epoch))
        for e in range(1, epoch + 1):

            vl, val_loss = self.get_validation_loss() if validate else -1, None
            for idx, (x, y) in enumerate(self.train_loader):

                if self.exit.exit_now:
                    self.save_state(self.path_save, self.fn_save)

                pbar.update(n=x.shape[0])

                x = x.to(self.device)
                y = y.to(self.device)

                logits, _, _ = self.model(x)

                pred = logits.view(-1, logits.shape[2])
                true = y.view(-1)

                train_loss = self.loss(pred, true)

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                if idx % log_interval or idx == 0:
                    tl = train_loss.data.mean()

                    self.train_losses[pbar.n] = tl
                    self.val_losses[pbar.n] = vl

                    pbar.set_postfix_str(
                        f"epoch: {e}/{epoch} , lr: {self.get_lr()} , train_loss: {tl:5f} , val_loss: {vl:5f}")

            # update the slope(a) according to the papers desc. schedule
            # self.slope_scheduler.step()

        # TODO some bug here epoch is not written as filename
        self.epoch += epoch

    def to(self, device: torch.device):
        self.device = device

        self.model.to(self.device)
        self.loss.to(self.device)

    def plot_loss(self, figsize: Tuple[float, float] = (10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(list(self.train_losses.keys()), list(self.train_losses.values()), linewidth=0.5, alpha=0.7,
                label="training loss")
        ax.plot(list(self.val_losses), list(self.val_losses.values()), linewidth=0.5, alpha=0.7,
                label="validation loss")
        ax.set_xlabel("samples used for training")
        ax.set_ylabel("mean cross entropy loss")
        ax.legend()
        plt.show()

    def save_state(self, path: str, filename: str):
        Path(path).mkdir(parents=True, exist_ok=True)

        # TODO save state with model description
        # TODO save state generate filename
        torch.save({"epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses
                    }, path + filename)

    def load_state(self, path: str):

        # TODO load state with model descripiton
        # TODO load state based on model parameters
        if Path(path).exists():
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
        else:
            print(f"file {path} not existing")

    def predict(self, encoded_text: torch.Tensor, k: int = 1) -> torch.Tensor:

        self.model.eval()
        logits, h, z = self.model(encoded_text.to(self.device))

        # TODO why?
        logits = logits[:, -1]
        sample = Multinomial(k, logits=logits).sample()
        prediction = sample.argmax().reshape((encoded_text.shape[0],))

        return prediction, h, z

    def sample_text(self, text: str, length: int, k: int = 1, online=True) -> str:
        print("we are sampling on", self.device, end="\n", flush=True)
        text = self.dataset.preprocess_text(text)
        T = len(text)

        encoded_text = self.dataset.encode(text).reshape((1, T, 1)).to(self.device)

        if online:
            print(text, end="")

        for i in range(length):
            prediction, h, z = self.predict(encoded_text, k)
            encoded_text = torch.cat((encoded_text, prediction.reshape(1, 1, 1)), dim=1)

            if online:
                print(self.dataset.decode(prediction), end="")

        return self.dataset.decode(encoded_text.squeeze()), h, z

    @staticmethod
    def print_cuda_info():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"cuda device: {i} / "
                      f"name: {torch.cuda.get_device_name(i)} / "
                      f"cuda-capability: {torch.cuda.get_device_capability(i)} / "
                      f"memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)} GB")
        else:
            print("cuda is not available")


class GracefulExit:
    exit_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.exit_now = True
