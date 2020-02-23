from typing import List, Tuple

import torch
import torch.distributions
from torch import nn

from hmlstm import HMLSTMCell, HMLSTMState, HMLSTMStatesList


class HMLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], bias: List[bool] = None):
        super(HMLSTM, self).__init__()

        self.num_layers = len(hidden_sizes)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.bias = [True for _ in hidden_sizes] if bias is None else bias

        self.cells = nn.ModuleList()

        for l, hidden_size in enumerate(hidden_sizes):
            hidden_size_top = 1 if l + 1 == self.num_layers else hidden_sizes[l + 1]
            hidden_size_bottom = input_size if l == 0 else hidden_sizes[l - 1]

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
        batch, seq_len, input_size = input.size()

        # states per layer at one element in the seq
        states = [self.init_state(batch, l=l) for l in range(self.num_layers)] if states is None else states

        hidden_states = []
        z_states = []

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
            h_layers = torch.cat([state.h for state in states], dim=1)
            z_layers = torch.cat([state.z for state in states], dim=1)

            hidden_states.append(h_layers)
            z_states.append(z_layers)

            # hidden_states.append([state.h for state in states])
            # z_states.append([state.z for state in states])

        h_all = torch.stack(hidden_states, dim=1)
        z_all = torch.stack(z_states, dim=1)

        # return hidden_states, z_states
        return h_all, z_all
