from datetime import datetime
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader

from hmlstm import SlopeScheduler
from .utils import EMAEarlyStopping as EarlyStopping, is_notebook

if is_notebook():
    from tqdm.notebook import tqdm as tqdm
else:
    from tqdm import tqdm


# TODO change trainer class so its properly supports continuing of training
class _Trainer(object):
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 valid_loader: DataLoader, loss: nn.Module = nn.CrossEntropyLoss(reduction="mean"),
                 device: torch.device = "cpu", path: str = "./"):

        # TODO implement graceful exit
        # self.exit = GracefulExit()
        self.early_stopping = EarlyStopping(checkpoint=self.save_state)
        self.path = path + datetime.now().strftime("%Y%m%d_%H%M%S") + "/"

        self.device = device
        self.model = model
        self.loss = loss

        # TODO maybe add optimizer as init parameter
        self.optimizer = torch.optim.Adam(model.parameters())

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.train_losses = {}
        self.valid_losses = {}

    def get_val_loss(self, val_loader: DataLoader, any: bool = False):

        self.model.eval()

        with torch.no_grad():
            valid_losses = []
            # x, y = iter(val_loader).next()

            if not any:
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    # TODO find better generic way
                    (out) = self.model(x)
                    logits = out[0]

                    valid_loss = self.loss(logits.view(-1, logits.shape[2]), y.view(-1))

                    valid_losses.append(valid_loss)
            else:
                x, y = iter(val_loader).next()

                x = x.to(self.device)
                y = y.to(self.device)

                # TODO find better generic way
                (out) = self.model(x)
                logits = out[0]

                valid_loss = self.loss(logits.view(-1, logits.shape[2]), y.view(-1))

                valid_losses.append(valid_loss)

        self.model.train()

        losses = torch.stack(valid_losses)

        return torch.mean(losses)

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def to(self, device: torch.device):
        self.device = device

        self.model.to(self.device)
        self.loss.to(self.device)

    def plot_loss(self, figsize: Tuple[float, float] = (10, 8), filename: str = None):
        plt.clf()

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(list(self.train_losses.keys()), list(self.train_losses.values()), linewidth=0.5, alpha=0.7,
                label="training loss")
        ax.plot(list(self.valid_losses.keys()), list(self.valid_losses.values()), linewidth=0.5, alpha=0.7,
                label="validation loss")
        ax.set_xlabel("number of updates")
        ax.set_ylabel("mean cross entropy loss")
        ax.legend()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        plt.clf()

    def save_state(self, name: str = "state", save_summary: bool = True, save_plot: bool = True):
        Path(self.path).mkdir(parents=True, exist_ok=True)

        filename = name + ".pt"

        torch.save({"model_summary": str(self.model),
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "train_losses": self.train_losses,
                    "val_losses": self.valid_losses
                    }, self.path + filename)

        if save_summary:
            with open(self.path + name + ".txt", "wt") as text_file:
                last_train_loss = -1 if not self.train_losses else self.train_losses[
                    sorted(self.train_losses.keys())[-1]]
                last_valid_loss = -1 if not self.valid_losses else self.valid_losses[
                    sorted(self.valid_losses.keys())[-1]]

                text_file.write("last train_loss: " + str(last_train_loss) + "\n")
                text_file.write("last valid_loss: " + str(last_valid_loss) + "\n")
                text_file.write("lr: " + str(self.get_lr()) + "\n")
                text_file.write("batch_size: " + str(self.train_loader.batch_size) + "\n")
                text_file.write("samples: " + str(len(self.train_loader.dataset)) + "\n")
                text_file.write("updates: " + str(len(self.train_losses)) + "\n")
                text_file.write("\n")
                text_file.write(str(self.model))

                text_file.close()

        if save_plot:
            self.plot_loss(filename=(self.path + name + ".png"))

    def load_state(self, path: str, print_summary: bool = False, load_optimizer: bool = False):

        if Path(path).exists():
            checkpoint = torch.load(path)

            if print_summary:
                model_summary = checkpoint["model_summary"]
                print("model summary...")
                print(model_summary)

            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.train_losses = checkpoint["train_losses"]
            self.valid_losses = checkpoint["val_losses"]
        else:
            print(f"file {path} not existing")


class LSTMTrainer(_Trainer):
    def __init__(self, model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
                 loss: nn.Module = nn.CrossEntropyLoss(reduction="mean"), device: torch.device = "cpu",
                 path: str = "./"):
        super().__init__(model, train_loader, valid_loader, loss, device, path)

    def train(self, epochs: int, lr: int = 0.01, log_interval: int = 50, validate: bool = False):
        samples = len(self.train_loader.dataset)
        batch_size = self.train_loader.batch_size
        updates = int((samples * epochs) / batch_size) + 1

        self.to(self.device)
        self.set_lr(lr)

        self.train_losses = {}
        self.valid_losses = {}

        pbar = tqdm(total=updates)

        self.model.train()
        for e in range(1, epochs + 1):
            for idx, (x, y) in enumerate(self.train_loader):

                x = x.to(self.device)
                y = y.to(self.device)

                (out) = self.model(x)
                logits = out[0]

                pred = logits.view(-1, logits.shape[2])
                true = y.view(-1)

                train_loss = self.loss(pred, true)
                tl = train_loss.data
                self.train_losses[pbar.n] = tl

                if not idx % log_interval:
                    valid_losses = self.get_val_loss(self.valid_loader) if validate else None
                    vl = valid_losses.data if validate else -1
                    self.valid_losses[pbar.n] = vl

                    if self.early_stopping(vl):
                        return vl
                else:
                    self.valid_losses[pbar.n] = vl

                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                pbar.set_postfix_str(
                    f"idx: {idx} , epoch: {e}/{epochs} , lr: {self.get_lr()} , patience: {self.early_stopping.patience} , train_loss: {tl:5f} , val_loss: {vl:5f}")
                pbar.update()

        pbar.close()

        self.save_state("complete")

        return self.get_val_loss(self.valid_loader)


class HMLSTMTrainer(_Trainer):

    def __init__(self, model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader,
                 loss: nn.Module = nn.CrossEntropyLoss(reduction="mean"), device: torch.device = "cpu",
                 path: str = "./", slope_factor: int = 25, checkpoint_th: float = 1.25):
        super(HMLSTMTrainer, self).__init__(model, train_loader, valid_loader, loss, device, path)

        self.slope_scheduler = SlopeScheduler(model.state_dict(), factor=slope_factor)
        self.early_stopping = EarlyStopping(checkpoint=self.save_state, threshold=checkpoint_th)

    def train(self, epochs: int, lr: int = 0.01, log_interval: int = 50, validate: bool = False,
              validation_th: float = 1.5):
        samples = len(self.train_loader.dataset)
        batch_size = self.train_loader.batch_size
        updates = int((samples * epochs) / batch_size) + 1

        self.to(self.device)
        self.set_lr(lr)
        self.model.train()

        self.train_losses = {}
        self.valid_losses = {}

        pbar = tqdm(total=updates)
        for e in range(1, epochs + 1):
            for idx, (x, y) in enumerate(self.train_loader):

                x = x.to(self.device)
                y = y.to(self.device)

                (out) = self.model(x)
                logits = out[0]

                pred = logits.view(-1, logits.shape[2])
                true = y.view(-1)

                train_loss = self.loss(pred, true)
                tl = train_loss.data

                self.train_losses[pbar.n] = tl

                if idx == 0 or (not idx % log_interval and tl < validation_th):
                    val_loss = self.get_val_loss(self.valid_loader) if validate else None
                    vl = val_loss.data if validate else -1
                    self.valid_losses[pbar.n] = vl

                    if self.early_stopping(vl):
                        return vl
                else:
                    self.valid_losses[pbar.n] = vl

                self.optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                pbar.set_postfix_str(
                    f"epoch: {e}/{epochs} , lr: {self.get_lr()} , patience: {self.early_stopping.patience} , slope: {self.slope_scheduler.get_slope():5f} , train_loss: {tl:5f} , val_loss: {vl:5f}")
                pbar.update()

                self.slope_scheduler.step()

        pbar.close()

        self.save_state("complete")

        return self.get_val_loss(self.valid_loader)
