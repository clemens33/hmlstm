# TODO add torchscript jit optimiziations
from typing import List, Any, Tuple

import torch

from torch import nn, jit

from lstm.cell import LayerNormLSTMCell


class LayerNormLSTM(torch.jit.ScriptModule):
    def __init__(self, input_size: int, hidden_sizes: List[int]) -> None:
        super(LayerNormLSTM, self).__init__()

        self.input_size = input_size
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes

        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            hidden_size_before = input_size if l == 0 else hidden_sizes[l - 1]

            self.cells.append(LayerNormLSTMCell(hidden_size_before, hidden_sizes[l]))

    # TODO maybe add a version where a input state can be passed into and are returned
    @torch.jit.script_method
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_size = input.size()

        h_prev = [torch.zeros(batch_size, hidden_size, device=input.device) for hidden_size in self.hidden_sizes]
        c_prev = [torch.zeros(batch_size, hidden_size, device=input.device) for hidden_size in self.hidden_sizes]

        hidden_states = []
        cell_states = []

        for t in range(seq_len):

            h = input[:, t]
            for l, cell in enumerate(self.cells):
                state = (h_prev[l], c_prev[l])

                h, c = cell(h, state)

                h_prev[l] = h
                c_prev[l] = c

            hidden_states += [h]
            cell_states += [c]

        return torch.stack(hidden_states, dim=1), torch.stack(cell_states, dim=1)


class LayerLSTM(jit.ScriptModule):
    def __init__(self, input_size: int, hidden_sizes: List[int]) -> None:
        super(LayerLSTM, self).__init__()

        self.input_size = input_size
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes

        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            hidden_size_before = input_size if l == 0 else hidden_sizes[l - 1]

            self.cells.append(nn.LSTMCell(hidden_size_before, hidden_sizes[l]))

    # TODO maybe add a version where a input state can be passed into and are returned
    @torch.jit.script_method
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_size = input.size()

        jit.annotate()

        h_prev = [torch.zeros(batch_size, hidden_size, device=input.device) for hidden_size in self.hidden_sizes]
        c_prev = [torch.zeros(batch_size, hidden_size, device=input.device) for hidden_size in self.hidden_sizes]

        hidden_states = []
        cell_states = []

        for t in range(seq_len):

            h = input[:, t]
            for l, cell in enumerate(self.cells):
                state = (h_prev[l], c_prev[l])

                h, c = cell(h, state)

                h_prev[l] = h
                c_prev[l] = c

            hidden_states += [h]
            cell_states += [c]

        return torch.stack(hidden_states, dim=1), torch.stack(cell_states, dim=1)
