import math
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import Parameter, init

from hmlstm import HMLSTMState, HardSigm, Round


class _HMLSTMCell(nn.Module):
    _num_chunks = 4

    def __init__(self, input_bottom_size: int, hidden_size: int, input_top_size: int, bias: bool = True) -> None:
        super(_HMLSTMCell, self).__init__()

        self.input_bottom_size = input_bottom_size
        self.hidden_size = hidden_size
        self.input_top_size = input_top_size
        self.bias = bias

        self.calc_z = _CalcZ()

    def reset_parameters(self):
        sd = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            if p.requires_grad:
                init.uniform_(p, -sd, sd)

    def reset_parameters2(self):
        for p in self.parameters():
            if p.requires_grad and p.ndim > 1:

                init.kaiming_uniform_(p)


class HMLSTMCell1(_HMLSTMCell):
    """
    hmlstm cell version which resembles the original paper the most but is least optimized
    idea is really to calculate based on the corresponding boundaries for each sample in the batch the appropriate
    operation. Unfortunately this turns out to be slow.
    """

    def __init__(self, input_bottom_size: int, hidden_size: int, input_top_size: int, bias: bool = True):
        super(HMLSTMCell1, self).__init__(input_bottom_size, hidden_size, input_top_size, bias)

        self.U = nn.ParameterList()
        self.R = nn.ParameterList()
        self.W = nn.ParameterList()

        [self.U.append(Parameter(torch.Tensor(input_top_size, hidden_size))) for _ in range(self._num_chunks)]
        [self.R.append(Parameter(torch.Tensor(hidden_size, hidden_size))) for _ in range(self._num_chunks)]
        [self.W.append(Parameter(torch.Tensor(input_bottom_size, hidden_size))) for _ in range(self._num_chunks)]

        self.U.append(Parameter(torch.Tensor(input_top_size, 1)))
        self.R.append(Parameter(torch.Tensor(hidden_size, 1)))
        self.W.append(Parameter(torch.Tensor(input_bottom_size, 1)))

        if bias:
            self.b = nn.ParameterList()
            [self.b.append(Parameter(torch.Tensor(hidden_size, ))) for _ in range(self._num_chunks)]
            self.b.append(Parameter(torch.Tensor(1, )))
        else:
            [self.register_parameter("b." + i, None) for i in range(self._num_chunks + 1)]

        self.reset_parameters()

    # TODO rewrite to single matrix for R, W, and U including samples indexing for diff operations
    def calc_gates(self, input: HMLSTMState, idx: torch.Tensor, gates: List[int] = [0, 1, 2, 3, 4]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # not necassary for the model (index removes zero samples) - but for the backward pass z needs be used!!!
        hb = input.h_bottom[idx, :] * input.z_bottom[idx, :]
        ht = input.h_top[idx, :] * input.z[idx, :]

        s_bottomup = [hb @ self.W[g] if g in gates else None for g in range(self._num_chunks + 1)]
        s_recurrent = [input.h[idx, :] @ self.R[g] if g in gates else None for g in range(self._num_chunks + 1)]
        s_topdown = [ht @ self.U[g] if g in gates else None for g in range(self._num_chunks + 1)]

        s = [s_bottomup[g] + s_recurrent[g] + s_topdown[g] + self.b[g] if g in gates else None for g in
             range(self._num_chunks + 1)]

        i = torch.sigmoid(s[0]) if 0 in gates else None
        g = torch.tanh(s[1]) if 1 in gates else None
        o = torch.sigmoid(s[2]) if 2 in gates else None
        f = torch.sigmoid(s[3]) if 3 in gates else None
        sz = torch.sigmoid(s[4]) if 4 in gates else None

        return i, g, o, f, sz

    def calc_state(self, input: HMLSTMState, flush_idx: torch.Tensor, copy_idx: torch.Tensor,
                   update_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # c = torch.empty(input.c.size())
        # h = torch.empty(input.h.size())
        c = input.c
        h = input.h

        if len(flush_idx) > 0:
            h[flush_idx, :], c[flush_idx, :] = self.flash(input, flush_idx)

        if len(update_idx) > 0:
            h[update_idx, :], c[update_idx, :] = self.update(input, update_idx)

        if len(copy_idx) > 0:
            h[copy_idx, :], c[copy_idx, :] = self.copy(input, copy_idx)

        return h, c

    def update(self, input: HMLSTMState, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i, g, o, f, _ = self.calc_gates(input, idx, [0, 1, 2, 3])

        c = input.c[idx, :] * f + i * g
        h = torch.tanh(c) * o

        return h, c

    def copy(self, input: HMLSTMState, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return input.h[idx, :], input.c[idx, :]

    def flash(self, input: HMLSTMState, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i, g, o, _, _ = self.calc_gates(input, idx, [0, 1, 2])

        c = i * g
        h = torch.tanh(c) * o

        return h, c

    def forward(self, input: HMLSTMState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        flush_idx = torch.eq(input.z, 1).nonzero()[:, 0]
        copy_idx = (torch.eq(input.z, 0) & torch.eq(input.z_bottom, 0)).nonzero()[:, 0]
        update_idx = (torch.eq(input.z, 0) & torch.eq(input.z_bottom, 1)).nonzero()[:, 0]

        h, c = self.calc_state(input, flush_idx, copy_idx, update_idx)

        _, _, _, _, sz = self.calc_gates(input, torch.arange(len(input.z)), [4])
        z = self.calc_z(sz)

        return h, c, z


class HMLSTMCell2(_HMLSTMCell):
    """
    fastest version (what I tried of the hmlstm cell)
    """

    def __init__(self, input_bottom_size: int, hidden_size: int, input_top_size: int, bias: bool = True):
        super(HMLSTMCell2, self).__init__(input_bottom_size, hidden_size, input_top_size, bias)

        self.U = Parameter(torch.Tensor(input_top_size, (self._num_chunks * hidden_size) + 1))
        self.R = Parameter(torch.Tensor(hidden_size, (self._num_chunks * hidden_size) + 1))
        self.W = Parameter(torch.Tensor(input_bottom_size, (self._num_chunks * hidden_size) + 1))

        if bias:
            self.b = Parameter(torch.Tensor(((self._num_chunks * hidden_size) + 1), ))
        else:
            self.register_parameter('b', None)

        self.reset_parameters()

    def calc_gates(self, input: HMLSTMState):

        s_bottomup = (input.h_bottom * input.z_bottom) @ self.W
        # s_bottomup = input.h_bottom @ self.W
        s_recurrent = input.h @ self.R
        s_topdown = (input.h_top * input.z) @ self.U
        # s_topdown = input.h_top @ self.U

        s = s_bottomup + s_recurrent + s_topdown + self.b

        split_sections = [self.hidden_size if i < self._num_chunks else 1 for i in range(self._num_chunks + 1)]
        si, sg, so, sf, sz = torch.split(s, split_sections, dim=1)

        i = torch.sigmoid(si)
        g = torch.tanh(sg)
        o = torch.sigmoid(so)
        f = torch.sigmoid(sf)

        return i, g, o, f, sz

    def forward(self, input: HMLSTMState):

        i, g, o, f, sz = self.calc_gates(input)

        c = torch.where(
            torch.eq(input.z, 1),
            i * g,  # flush
            torch.where(
                torch.eq(input.z_bottom, 0),
                input.c,  # copy
                input.c * f + i * g  # update
            )
        )

        h = torch.where(
            torch.eq(input.z, 0) & torch.eq(input.z_bottom, 0),
            input.h,  # copy
            torch.tanh(c) * o  # update / flash
        )

        z = self.calc_z(sz)

        return h, c, z


class HMLSTMCell3(_HMLSTMCell):
    """
    another version of the hmlstm cell - not quite as fast - doesn't use torch where or indexing
    """

    def __init__(self, input_bottom_size: int, hidden_size: int, input_top_size: int, bias: bool = True, a: int = 7,
                 th: float = 0.5):
        super(HMLSTMCell3, self).__init__(input_bottom_size, hidden_size, input_top_size, bias)

        self.U = Parameter(torch.Tensor(input_top_size, (self._num_chunks * hidden_size) + 1))
        self.R = Parameter(torch.Tensor(hidden_size, (self._num_chunks * hidden_size) + 1))
        self.W = Parameter(torch.Tensor(input_bottom_size, (self._num_chunks * hidden_size) + 1))

        if bias:
            self.b = Parameter(torch.Tensor(((self._num_chunks * hidden_size) + 1), ))
        else:
            self.register_parameter('b', None)

        self.reset_parameters()

    def calc_gates(self, input: HMLSTMState):

        s_bottomup = (input.h_bottom * input.z_bottom) @ self.W
        s_recurrent = input.h @ self.R
        s_topdown = (input.h_top * input.z) @ self.U

        s = s_bottomup + s_recurrent + s_topdown + self.b

        split_sections = [self.hidden_size if i < self._num_chunks else 1 for i in range(self._num_chunks + 1)]
        si, sg, so, sf, sz = torch.split(s, split_sections, dim=1)

        i = torch.sigmoid(si)
        g = torch.tanh(sg)
        o = torch.sigmoid(so)
        f = torch.sigmoid(sf)

        return i, g, o, f, sz

    def forward(self, input: HMLSTMState):

        i, g, o, f, sz = self.calc_gates(input)

        # one = torch.scalar_tensor(1, device=input.z_bottom.device)

        c = input.z * (i * g) + (1 - input.z) * (1 - input.z_bottom) * input.c + (
                1 - input.z) * input.z_bottom * (f * input.c + i * g)
        h = input.z * o * torch.tanh(c) + (1 - input.z) * (1 - input.z_bottom) * input.h + (
                1 - input.z) * input.z_bottom * o * torch.tanh(
            c)

        # z = torch.ones(input.z.size(), device=input.z.device)
        z = self.calc_z(sz)

        return h, c, z


class HMLSTMCell4(HMLSTMCell2):
    """
    layer norm version of the hmlstm cell

    idea based on the official pytorch fast rnn benchmark lstm cell layer norm implementation
    https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py
    I'm not sure why they add layernorm for the recurrent and the incoming pre activations
    """

    def __init__(self, input_bottom_size: int, hidden_size: int, input_top_size: int, bias: bool = True):
        super(HMLSTMCell4, self).__init__(input_bottom_size, hidden_size, input_top_size, bias)

        # self.ln_bottomup = nn.LayerNorm(self._num_chunks * hidden_size + 1)
        self.ln_bottomup = nn.Identity()

        # self.ln_recurent = nn.LayerNorm(self._num_chunks * hidden_size + 1)
        self.ln_recurent = nn.Identity()

        # TODO check - if layer norm is applied to the top down connection separately gradients are nan ???
        # self.ln_topdown = nn.LayerNorm(self._num_chunks * hidden_size + 1)
        self.ln_topdown = nn.Identity()

        self.ln_s = nn.LayerNorm(self._num_chunks * hidden_size + 1)
        # self.ln_s = nn.Identity()

        self.ln_c = nn.LayerNorm(hidden_size)
        # self.ln_c = nn.Identity()

    def calc_gates(self, input: HMLSTMState):
        s_bottomup = self.ln_bottomup((input.h_bottom * input.z_bottom) @ self.W)
        s_recurrent = self.ln_recurent(input.h @ self.R)
        s_topdown = self.ln_topdown((input.h_top * input.z) @ self.U)

        s = self.ln_s(s_bottomup + s_recurrent + s_topdown) + self.b

        split_sections = [self.hidden_size if i < self._num_chunks else 1 for i in range(self._num_chunks + 1)]
        si, sg, so, sf, sz = torch.split(s, split_sections, dim=1)

        i = torch.sigmoid(si)
        g = torch.tanh(sg)
        o = torch.sigmoid(so)
        f = torch.sigmoid(sf)

        return i, g, o, f, sz

    def forward(self, input: HMLSTMState):
        i, g, o, f, sz = self.calc_gates(input)

        c = torch.where(
            torch.eq(input.z, 1),
            self.ln_c(i * g),  # flush
            torch.where(
                torch.eq(input.z_bottom, 0),
                input.c,  # copy
                self.ln_c(input.c * f + i * g)  # update
            )
        )

        h = torch.where(
            torch.eq(input.z, 0) & torch.eq(input.z_bottom, 0),
            input.h,  # copy
            torch.tanh(c) * o  # update / flash
        )

        z = self.calc_z(sz)

        return h, c, z


class _CalcZ(nn.Module):
    def __init__(self, a: int = 1, th: float = 0.5):
        super(_CalcZ, self).__init__()

        self.round = Round(th)
        self.hardsigm = HardSigm(a)

    def forward(self, sz: torch.Tensor) -> torch.Tensor:
        z_tilde = self.hardsigm(sz)
        z = self.round(z_tilde)

        #z = torch.sigmoid(sz)

        return z
