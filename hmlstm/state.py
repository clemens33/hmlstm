from dataclasses import dataclass
from typing import List

import torch


@dataclass
class HMLSTMState:
    h_bottom: torch.Tensor = None  # at the same time step
    h_top: torch.Tensor = None  # at the last time step
    h: torch.Tensor = None
    c: torch.Tensor = None
    z_bottom: torch.Tensor = None
    z: torch.Tensor = None
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.to(self.device)


    def to(self, device: torch.device = "cpu"):
        for k, t in self.__dict__.items():
            if isinstance(t, torch.Tensor):
                self.__dict__[k] = t.to(device)


# TODO better state handling and returns in network.py and model.py
class HMLSTMStatesList(list):
    def __init__(self, states=None):
        list.__init__(self, states)

    def get_h(self, layer: int = 0, seq: int = 0) -> torch.Tensor:
        return self[seq][layer].h

    def get_z(self, layer: int = 0, seq: int = 0) -> torch.Tensor:
        return self[seq][layer].z

    def get_hs(self) -> List[torch.Tensor]:
        for s in len(self):
            hs = [state.h for state in s]

        return hs
