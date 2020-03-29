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
from utils import HMLSTMTrainer as Trainer, print_cuda_info
from hmlstm import HMLSTMNetwork, utils as hmlstm_utils

# %%


# %%

use_cuda = torch.cuda.is_available()
#use_cuda = False

if use_cuda:
    print_cuda_info()
    torch.cuda.empty_cache()
    #torch.set_num_threads(1)

device = torch.device('cuda' if use_cuda else 'cpu')

project_name = "shakespeare"

seq_length = 100

path_data = path + "projects/" + project_name + "/data/"
path_states = path + "projects/" + project_name + "/states/"
dataset = TextDataset(path_data + "data.txt", seq_length)

train_dataset, valid_dataset = dataset.split()

bsv = int(len(valid_dataset)/4) + 1


batch_size_train = 1000
batch_size_valid = bsv
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=use_cuda)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=True, pin_memory=use_cuda)

# %%

input_size = 1
embedding_size_input = 32
hidden_sizes = [256, 128, 128]

embedding_size_output = 256
linear_sizes = [128]
output_size = dataset.vocab_len()

model = HMLSTMNetwork(input_size, embedding_size_input, hidden_sizes, embedding_size_output, linear_sizes, output_size, layer_norm=True)
trainer = Trainer(model, train_loader, valid_loader, device=device, path=path_states, slope_factor=100, checkpoint_th=1.3)

# %%

epochs = 30

trainer.to(device)
vl = trainer.train(epochs=epochs, lr=0.002, log_interval=5, validate=True, validation_th=1.35)

trainer.plot_loss()

print(vl)

# %%
