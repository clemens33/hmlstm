# %%
import sys

import torch
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
from utils import LSTMTrainer as Trainer, print_cuda_info
from lstm.network import BaselineLSTM

# %%


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
#batch_size_valid = int(len(valid_dataset)/4) + 1
batch_size_valid = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=use_cuda)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=use_cuda)

# %%

input_size = 1
embedding_size = 77
hidden_sizes = [256, 255, 256]
linear_sizes = []
output_size = dataset.vocab_len()


model = BaselineLSTM(input_size, embedding_size, hidden_sizes, output_size, linear_sizes, layer_norm=False)
trainer = Trainer(model, train_loader, valid_loader, device=device, path=path_states)

# %%

epochs = 10

trainer.to(device)
vl = trainer.train(epochs=epochs, lr=0.001527, log_interval=1, validate=False)

trainer.plot_loss()

print(vl)

# %%
