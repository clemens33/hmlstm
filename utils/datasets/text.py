import math
import re
import string
from pathlib import PosixPath
from typing import Tuple

import torch
from torch.utils.data import Dataset, random_split


class _Characters(Dataset):
    _vocabulary = None

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

        return x, y

    def split(self, val_split_factor: float = .2):
        n = len(self)
        val_length = int(n * val_split_factor)

        lengths = [n - val_length, val_length]

        return random_split(self, lengths)

    def preprocess_text(self, raw_text: str, lower_case: bool = False) -> str:
        text = re.sub("[^" + self._vocabulary + "]+", "", raw_text)

        text = self.lower(text) if lower_case else text

        return text

    def lower(self, text):
        self._vocabulary = self._vocabulary.lower()

        return text.lower()

    def encode(self, text: str, device: torch.device = "cpu") -> torch.Tensor:
        encoded_text = [self._vocabulary.index(char) for char in text]

        return torch.Tensor(encoded_text).long().to(device)

    def decode(self, encoded_text: torch.Tensor) -> str:
        decoded_text = "".join(self._vocabulary[idx] for idx in encoded_text.tolist())

        return decoded_text


class CharactersAll(_Characters):
    _vocabulary = string.ascii_letters + string.digits + string.punctuation + string.whitespace

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)


class CharactersPunctuation(_Characters):
    _vocabulary = string.ascii_letters + string.digits + string.punctuation + " "

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)


class CharactersText(_Characters):
    _vocabulary = string.ascii_letters + string.digits + " _,.!?"

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)


class CharactersLower(_Characters):
    _vocabulary = string.ascii_lowercase + " :,.!?"

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        super().__init__(text_file, seq_length, step_size, device)

    def preprocess_text(self, raw_text: str, lower_case: bool = False) -> str:
        return super().preprocess_text(raw_text, True)


class Characters(_Characters):
    _vocabulary = ""
    _blacklist = ""

    def __init__(self, text_file: PosixPath, seq_length: int, step_size: int = None, device: torch.device = "cpu"):
        self.device = device
        self.seq_length = seq_length
        self.step_size = seq_length if step_size is None else step_size

        with open(text_file, encoding="utf8", mode="r") as file:
            raw_text = file.read()
            self._vocabulary += self.build_vocabulary(raw_text)
            text = self.preprocess_text(raw_text)
            file.close()

        encoded_text = self.encode(text)
        self.data = encoded_text.to(self.device)

    def build_vocabulary(self, raw_text: str):
        return ''.join(set(raw_text))

    def preprocess_text(self, raw_text: str, lower_case: bool = False) -> str:
        # removes characters from text
        text = re.sub(self._blacklist, '', raw_text)

        text = super().lower(text) if lower_case else text

        return text
