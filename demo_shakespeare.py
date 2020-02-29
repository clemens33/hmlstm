# %%
import sys
from datetime import datetime
from typing import List, Callable, Tuple

import torch
from torch import nn, jit
from torch.utils.data import DataLoader

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
from utils._trainer import _Trainer
from lstm import LayerNormLSTM, LayerLSTM


# %%

class BaselineNetwork(nn.Module):
    def __init__(self, input_size: int, embedding_size: int, hidden_sizes: int, output_size: int,
                 linear_sizes: List[int] = None, layer_norm: bool = False):
        super(BaselineNetwork, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.linear_sizes = linear_sizes
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, embedding_size)

        if layer_norm:
            self.lstm = LayerNormLSTM(embedding_size, hidden_sizes)
        elif all(h == hidden_sizes[0] for h in hidden_sizes):
            # fastest version but no layer norm and no diff hidden sizes
            self.lstm = nn.LSTM(embedding_size, hidden_sizes[-1], self.num_layers, batch_first=True)
        else:
            self.lstm = LayerLSTM(embedding_size, hidden_sizes)

        self.fc_layers = nn.ModuleList()
        if linear_sizes is None or linear_sizes == []:
            self.fc = nn.Identity()
            self.output = nn.Linear(hidden_sizes[-1], output_size)
        else:
            for i, linear_size in enumerate(linear_sizes):
                linear_size_before = hidden_sizes[-1] if i == 0 else linear_sizes[i - 1]

                self.fc_layers.append(nn.Sequential(
                    nn.Linear(linear_size_before, linear_size),
                    nn.LayerNorm(linear_size) if layer_norm else nn.Identity(),
                    nn.Tanh()
                ))

            self.output = nn.Linear(linear_sizes[-1], output_size)

    def fc(self, input: torch.Tensor) -> torch.Tensor:
        out = input

        for fc_layer in self.fc_layers:
            out = fc_layer(out)

        return out

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """

        :param input: [batch_size, seq_len]
        :return:
        """
        emb = self.embedding(input)
        h, _ = self.lstm(emb)
        fc = self.fc(h)
        out = self.output(fc)

        # TODO change output remove nones
        return out, None, None


# %%

use_cuda = torch.cuda.is_available()
# use_cuda = False

if use_cuda:
    _Trainer.print_cuda_info()
    torch.cuda.empty_cache()

device = torch.device('cuda' if use_cuda else 'cpu')

project_name = "shakespeare"

seq_length = 100

path_data = path + "projects/" + project_name + "/data/"
dataset = TextDataset(path_data + "data.txt", seq_length)

train_dataset, valid_dataset = dataset.split()

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=True, pin_memory=use_cuda)

# %%

input_size = 1
embedding_size = 32
hidden_sizes = [512]
linear_sizes = []
output_size = len(dataset._vocabulary)

model = BaselineNetwork(input_size, embedding_size, hidden_sizes, output_size, linear_sizes, layer_norm=False)
loss = nn.CrossEntropyLoss()

trainer = _Trainer(model, train_loader, valid_loader, device=device)

# %%

epoch = 10

trainer.to(device)

# fn_save = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(trainer.epoch) + ".pt"
# path_states = path + "projects/" + project_name + "/states/"

trainer.train(epoch=epoch, lr=0.01, validate=False)

# trainer.save_state(path_states, fn_save)

# %%


# %%

# fn_load = "20200225_201022_0.pt"
# trainer.load_state(path_states + fn_load)

# fn_load = "20200225_202951_0.pt"

trainer.plot_loss()

# %%

trainer.to("cpu")
text = "Henry"
sampled_text, h, z = trainer.sample_text(text, length=100, k=3, online=True)
