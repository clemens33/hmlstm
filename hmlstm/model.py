from typing import List, Tuple

import torch
import torch.distributions
from torch import nn

from hmlstm import LayerNormHMLSTMCell, HMLSTMCell, HMLSTMState, HMLSTMStatesList


class HMLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], bias: List[bool] = None, layer_norm: bool = False):
        super(HMLSTM, self).__init__()

        self.num_layers = len(hidden_sizes)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.bias = [True for _ in hidden_sizes] if bias is None else bias
        self.layer_norm = layer_norm

        self.cells = nn.ModuleList()

        for l, hidden_size in enumerate(hidden_sizes):
            hidden_size_top = 1 if l + 1 == self.num_layers else hidden_sizes[l + 1]
            hidden_size_bottom = input_size if l == 0 else hidden_sizes[l - 1]

            if layer_norm:
                self.cells.append(LayerNormHMLSTMCell(hidden_size_bottom, hidden_size, hidden_size_top, self.bias[l]))
            else:
                self.cells.append(HMLSTMCell(hidden_size_bottom, hidden_size, hidden_size_top, self.bias[l]))

    def get_device(self) -> torch.device:
        return next(self.cells[0].parameters()).device

    def init_state(self, n: int, l: int = 0) -> HMLSTMState:
        hidden_size_top = 1 if l + 1 == self.num_layers else self.hidden_sizes[l + 1]
        hidden_size_bottom = self.input_size if l == 0 else self.hidden_sizes[l - 1]

        state = HMLSTMState(
            h_bottom=torch.zeros((n, hidden_size_bottom)),
            h_top=torch.zeros((n, hidden_size_top)),
            h=torch.zeros((n, self.hidden_sizes[l])),
            c=torch.zeros((n, self.hidden_sizes[l])),
            z_bottom=torch.ones((n, 1)),
            z=torch.zeros((n, 1)),
            device=self.get_device()
        )

        return state

    def forward(self, input: torch.Tensor, states: List[HMLSTMState] = None) -> Tuple[
        List[torch.Tensor], HMLSTMStatesList]:
        batch_size, seq_len, input_size = input.size()

        # states per layer at one element in the seq
        states = [self.init_state(batch_size, l=l) for l in range(self.num_layers)] if states is None else states

        hidden_states = torch.zeros(batch_size, seq_len, sum(self.hidden_sizes), device=input.device)
        z_states = torch.zeros(batch_size, seq_len, len(self.hidden_sizes), device=input.device)

        for t in range(seq_len):

            states[0].h_bottom = input[:, t]
            for l, cell in enumerate(self.cells):
                h, c, z = cell(states[l])

                states[l].h = h
                states[l].c = c
                states[l].z = z

                if l + 1 < self.num_layers:
                    states[l + 1].z_bottom = z
                    states[l + 1].h_bottom = h

                if l > 0:
                    states[l - 1].h_top = h

            # TODO check if cat places layers in reversed order
            hidden_states[:, t] = torch.cat([state.h for state in states], dim=1)
            z_states[:, t] = torch.cat([state.z for state in states], dim=1)

        return hidden_states, z_states



