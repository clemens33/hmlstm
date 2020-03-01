from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from .utils import SMAEarlyStopping, is_notebook


class Trainer(object):
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, loss: nn.Module = nn.CrossEntropyLoss(reduction="mean"),
                 device: torch.device = "cpu", path: str = "./"):

        # TODO implement graceful exit
        # self.exit = GracefulExit()
        self.stop_early = SMAEarlyStopping()
        self.path = path

        self.device = device
        self.model = model
        self.loss = loss

        # TODO maybe add optimizer as init parameter
        self.optimizer = torch.optim.Adam(model.parameters())

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_losses = {}
        self.val_losses = {}

    def get_val_loss(self, val_loader: DataLoader):

        self.model.eval()

        with torch.no_grad():
            val_losses = []
            # x, y = iter(val_loader).next()
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                # TODO find better generic way
                (out) = self.model(x)
                logits = out[0]

                val_loss = self.loss(logits.view(-1, logits.shape[2]), y.view(-1))

                val_losses.append(val_loss)

        self.model.train()

        losses = torch.stack(val_losses)

        return torch.mean(losses)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def train(self, epochs: int, lr: int = 0.01, log_interval: int = 50, validate: bool = False):
        samples = len(self.train_loader.dataset)
        batch_size = self.train_loader.batch_size
        updates = int((samples * epochs) / batch_size) + 1

        print(f"number of updates: {updates} / training on {samples * epochs} samples / training on {self.device} \n",
              flush=True)

        self.to(self.device)
        self.set_lr(lr)
        self.model.train()

        self.train_losses = {}
        self.val_losses = {}

        # TODO check notebook tqdm
        pbar = tqdm.notebook.tqdm(total=(updates)) if is_notebook() else tqdm.tqdm(total=(updates))

        for e in range(1, epochs + 1):
            for idx, (x, y) in enumerate(self.train_loader):

                x = x.to(self.device)
                y = y.to(self.device)

                (out) = self.model(x)
                logits = out[0]

                pred = logits.view(-1, logits.shape[2])
                true = y.view(-1)

                train_loss = self.loss(pred, true)

                if not idx % log_interval or idx == 0:
                    val_loss = self.get_val_loss(self.val_loader) if validate else None

                    tl = train_loss.data
                    vl = val_loss.data if validate else -1

                    self.train_losses[pbar.n] = tl
                    self.val_losses[pbar.n] = vl

                    pbar.set_postfix_str(
                        f"epoch: {e}/{epochs} , lr: {self.get_lr()} , train_loss: {tl:5f} , val_loss: {vl:5f}")

                    if self.stop_early(vl):
                        self.save_state(self.path,
                                        "vl_" + str(round(float(vl), 6)) + "_upd_" + str(pbar.n) + "_lr_" + str(
                                            self.get_lr()))

                        return vl

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                pbar.set_postfix_str(
                    f"epoch: {e}/{epochs} , lr: {self.get_lr()} , train_loss: {tl:5f} , val_loss: {vl:5f}")
                pbar.update()

        self.save_state(self.path,
                        "vl_" + str(round(float(vl), 6)) + "_upd_" + str(pbar.n) + "_lr_" + str(self.get_lr()))

        return vl

    def to(self, device: torch.device):
        self.device = device

        self.model.to(self.device)
        self.loss.to(self.device)

    def plot_loss(self, figsize: Tuple[float, float] = (10, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(list(self.train_losses.keys()), list(self.train_losses.values()), linewidth=0.5, alpha=0.7,
                label="training loss")
        ax.plot(list(self.val_losses.keys()), list(self.val_losses.values()), linewidth=0.5, alpha=0.7,
                label="validation loss")
        ax.set_xlabel("number of updates")
        ax.set_ylabel("mean cross entropy loss")
        ax.legend()
        plt.show()

    def save_state(self, path: str, postfix: str = ""):
        Path(path).mkdir(parents=True, exist_ok=True)

        filename = datetime.now().strftime("%H%M%S") + "_" + postfix + ".pt"

        torch.save({"model_summary": str(self.model),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_losses": self.train_losses,
                    "val_losses": self.val_losses
                    }, path + filename)

    def load_state(self, path: str):

        if Path(path).exists():
            checkpoint = torch.load(path)

            model_summary = checkpoint["model_summary"]
            print("model summary...")
            print(model_summary)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.train_losses = checkpoint["train_losses"]
            self.val_losses = checkpoint["val_losses"]
        else:
            print(f"file {path} not existing")
