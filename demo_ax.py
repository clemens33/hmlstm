# %%
import sys
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from ax.service.managed_loop import optimize

try:
    from google.colab import drive

    IN_COLAB = True

    drive.mount('/content/drive')
    path = "/content/drive/My Drive/Colab Notebooks/"

    # for python imports from google drive
    sys.path.append(path)
except:
    IN_COLAB = False
    path = "./"

from utils.datasets import Characters as TextDataset
from utils import Trainer, print_cuda_info, is_notebook
from models.baseline import BaselineNetwork

if is_notebook():
    from tqdm.notebook import tqdm_notebook as tqdm
    print("running in notebook and/or colab")
else:
    from tqdm import tqdm


# %%

def train(architecture, data_loader: DataLoader, parameters: Dict[str, float],
          device: torch.device) -> nn.Module:
    epochs = parameters.get("epochs", 1)
    embedding_size = parameters.get("embedding_size", 16)
    hidden_size = parameters.get("hidden_size", 128)
    lr = parameters.get("lr", 0.01)

    samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    updates = int((samples * epochs) / batch_size)

    print(f"epochs: {epochs} / embedding_size: {embedding_size} / hidden_size: {hidden_size} / lr: {lr:5f}")

    model = architecture(
        input_size=parameters.get("input_size", 1.0),
        embedding_size=embedding_size,
        hidden_sizes=[hidden_size],
        output_size=parameters.get("output_size", -1),
        linear_sizes=[],
        layer_norm=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr)

    pbar = tqdm(total=updates)


    model.to(device)
    criterion.to(device)

    model.train()

    for e in range(epochs):
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()

            (out) = model(inputs)
            logits = out[0]

            pred = logits.view(-1, logits.shape[2])
            true = labels.view(-1)

            loss = criterion(pred, true)
            loss.backward()
            optimizer.step()

            pbar.set_postfix_str(
                f"epoch: {e + 1}/{epochs} , train_loss: {loss:5f}")

            pbar.update()

    pbar.close()

    return model


# %%

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    criterion = nn.CrossEntropyLoss()

    model.eval()

    with torch.no_grad():
        losses = []
        # x, y = iter(val_loader).next()
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # TODO find better generic way
            (out) = model(inputs)
            logits = out[0]

            loss = criterion(logits.view(-1, logits.shape[2]), labels.view(-1))

            losses.append(loss)

    losses = torch.stack(losses)

    score = float(torch.mean(losses))

    print(f"validation loss: {score}")

    return score


# %%

use_cuda = torch.cuda.is_available()
# use_cuda = False

if use_cuda:
    print_cuda_info()
    torch.cuda.empty_cache()

device = torch.device('cuda' if use_cuda else 'cpu')

project_name = "shakespeare"

seq_length = 100

path_data = path + "projects/" + project_name + "/data/"
path_states = path + "projects/" + project_name + "/states/"
dataset = TextDataset(path_data + "data.txt", seq_length)

train_dataset, valid_dataset = dataset.split()

batch_size_train = 64
batch_size_valid = int(len(valid_dataset) / 4) + 1
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=use_cuda)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=use_cuda)


# %%

def train_evaluate(parameterization):
    model = train(BaselineNetwork, train_loader, parameters=parameterization, device=device)
    return evaluate(model, valid_loader, device)


# %%

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "input_size", "type": "fixed", "value": 1},
        {"name": "embedding_size", "type": "range", "bounds": [1, 64]},
        {"name": "hidden_size", "type": "range", "bounds": [8, 512]},
        {"name": "output_size", "type": "fixed", "value": len(dataset._vocabulary)},
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "epochs", "type": "fixed", "value": 5},
    ],
    evaluation_function=train_evaluate,
    objective_name='cross entropy loss',
    minimize=True
)

print(best_parameters)
