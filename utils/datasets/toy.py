from typing import Tuple

import torch
from torch.utils.data import Dataset


class Numbers(Dataset):

    def __init__(self, length: int, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        self.device = device
        self.length = length
        self.seq_length = seq_length
        self.step_size = seq_length if step_size is None else step_size

        self.data = self.generate_numbers()

    def generate_numbers(self) -> torch.Tensor:
        return torch.arange(0, self.length, device=self.device)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        N = self.data.shape[0]

        from_ptr = idx * self.step_size
        to_ptr = from_ptr + self.seq_length

        if to_ptr > (N - 2):
            from_ptr = N - self.seq_length - 1
            to_ptr = N - 1

        x = self.data[from_ptr:to_ptr]
        y = self.data[from_ptr + 1:to_ptr + 1]

        return x.reshape(self.seq_length, 1), y
