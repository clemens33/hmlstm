# TODO switch to torchscript for performance
from typing import List, Tuple

import torch
from torch import nn

from lstm import LayerNormLSTM, LayerLSTM


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

        :param input: [batch_size, seq_len, input_size]
        :return:
        """
        emb = self.embedding(input)
        h, _ = self.lstm(emb)
        fc = self.fc(h)
        out = self.output(fc)

        return out, None
