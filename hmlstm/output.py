from typing import List, Callable

import torch
from torch import nn
from torch.nn import Parameter


class HMLSTMOutput(nn.Module):
    def __init__(self, embedding_size: int, input_sizes: List[int], linear_sizes: List[int], output_size: int,
                 bias: List[bool] = None, layer_norm: bool = False):
        super(HMLSTMOutput, self).__init__()

        self.embedding_size = embedding_size
        self.input_sizes = input_sizes
        self.linear_sizes = linear_sizes
        self.output_size = output_size

        self.num_layers = len(input_sizes)
        self.num_linears = len(linear_sizes)
        self.bias = [True for _ in linear_sizes] if bias is None else bias
        self.layer_norm = layer_norm

        #self.w = nn.ParameterList()
        #[self.w.append(Parameter(torch.Tensor(sum(input_sizes), 1))) for _ in input_sizes]
        self.linear_gates = nn.ModuleList()
        for _ in input_sizes:
            if layer_norm:
                linear_gate = nn.Sequential(
                    nn.Linear(sum(input_sizes), 1, bias=False),
                    nn.LayerNorm(1)
                )
            else:
                linear_gate = nn.Linear(sum(input_sizes), 1, bias=False)

            self.linear_gates.append(linear_gate)

        self.embeddings = nn.ModuleList()
        [self.embeddings.append(nn.Linear(input_size, embedding_size)) for input_size in input_sizes]

        self.ln_embeddings = nn.LayerNorm(embedding_size) if layer_norm else nn.Identity()

        #self.activation = nn.SELU()
        self.activation = nn.ReLU()

        self.linears = nn.ModuleList()
        for i, linear_size in enumerate(linear_sizes):
            linear_size_before = embedding_size if i == 0 else linear_sizes[i - 1]

            if layer_norm:
                linear = nn.Sequential(
                    nn.Linear(linear_size_before, linear_size, self.bias[i]),
                    nn.LayerNorm(linear_size)
                )
            else:
                linear = nn.Linear(linear_size_before, linear_size, self.bias[i])

            self.linears.append(linear)

        if linear_sizes is None or linear_sizes == []:
            self.output = nn.Linear(embedding_size, output_size)
        else:
            self.output = nn.Linear(linear_sizes[-1], output_size)

    def fnn(self, input: torch.Tensor, activation: Callable = torch.tanh) -> torch.Tensor:
        out = input

        for linear in self.linears:
            s = linear(out)

            out = activation(s)

        return out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch, seq_len, input_size = input.size()

        out = torch.zeros(batch, seq_len, self.output_size, device=input.device)

        for t in range(seq_len):
            h = input[:, t]

            h_layers = torch.split_with_sizes(h, self.input_sizes, dim=1)

            s = torch.zeros(batch, self.embedding_size, device=h.device)
            for l, hl in enumerate(h_layers):
                #sg = h @ self.w[l]
                sg = self.linear_gates[l](h)
                g = torch.sigmoid(sg)

                s = s + self.embeddings[l](hl * g)

            s = self.ln_embeddings(s)
            he = self.activation(s)

            fnn = self.fnn(he)
            out[:, t] = self.output(fnn)

        return out
