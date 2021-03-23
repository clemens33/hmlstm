from typing import List, Tuple, Callable, Union

import torch
from torch import nn

from hmlstm import HMLSTM, HMLSTMOutput


class HMLSTMNetwork(nn.Module):
    # TODO hinting for embedding_size_input - int | List[int]
    def __init__(self, input_size: int, embedding_size_input: Union[int, List[int]], hidden_sizes: List[int],
                 embedding_size_output: int,
                 linear_sizes: List[int], output_size: int, layer_norm: bool = False):
        super(HMLSTMNetwork, self).__init__()

        self.num_layers = len(hidden_sizes)
        self.input_size = input_size
        self.embedding_size_input = embedding_size_input
        self.hidden_sizes = hidden_sizes
        self.embedding_size_output = embedding_size_output
        self.linear_sizes = linear_sizes
        self.output_size = output_size
        self.layer_norm = layer_norm

        # generic input layers
        if isinstance(embedding_size_input, list):
            self.embedding = nn.ModuleList()

            for i, embedding_size in enumerate(embedding_size_input):
                embedding_size_before = input_size if i == 0 else embedding_size_input[i - 1]

                self.embedding.append(nn.Linear(embedding_size_before, embedding_size))
        else:
            # input layer for index based embeddings (e.g. characters)
            self.embedding = nn.Embedding(output_size, embedding_size_input)

        self.hmlstm = HMLSTM(embedding_size_input, hidden_sizes, layer_norm=layer_norm)
        self.output = HMLSTMOutput(embedding_size_output, hidden_sizes, linear_sizes, output_size,
                                   layer_norm=layer_norm)

        # self.output = nn.Linear(hidden_sizes[-1], output_size)

    def fnn(self, input: torch.Tensor, activation: Callable = torch.tanh) -> torch.Tensor:
        out = input

        for linear in self.linears:
            s = linear(out)

            out = activation(s)

        return out

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        emb = self.embedding(input)
        h, z = self.hmlstm(emb)
        out = self.output(h)

        return out, (h, z)
