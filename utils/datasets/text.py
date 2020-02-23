import math
import re
import string
from pathlib import PosixPath
from typing import Tuple

import torch
from torch.utils.data import Dataset


class _Characters(Dataset):
    VOCABULARY = None

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        self.device = device
        self.seq_length = seq_length
        self.step_size = seq_length if step_size is None else step_size

        with open(text_file, encoding="utf8", mode="r") as file:
            raw_text = file.read()
            text = self.preprocess_text(raw_text)
            file.close()

        encoded_text = self.encode(text)
        self.data = encoded_text.to(self.device)

    def __len__(self) -> int:
        # N = self.data.shape[0] - self.seq_length
        N = self.data.shape[0]
        L = math.ceil(N / self.step_size)
        return L

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


    def preprocess_text(self, raw_text: str, lower_case: bool = False) -> str:
        text = re.sub("[^" + self.VOCABULARY + "]+", "", raw_text)
        text = text.lower() if lower_case else text

        return text

    def encode(self, text: str, device: torch.device = "cpu") -> torch.Tensor:
        encoded_text = [self.VOCABULARY.index(char) for char in text]

        return torch.Tensor(encoded_text).long().to(device)

    def decode(self, encoded_text: torch.Tensor) -> str:
        decoded_text = "".join(self.VOCABULARY[idx] for idx in encoded_text.tolist())

        return decoded_text


class CharactersAll(_Characters):
    VOCABULARY = string.ascii_letters + string.digits + string.punctuation + string.whitespace

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)


class CharactersPunctuation(_Characters):
    VOCABULARY = string.ascii_letters + string.digits + string.punctuation + " "

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)


class CharactersText(_Characters):
    VOCABULARY = string.ascii_letters + string.digits + " _,.!?"

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)
