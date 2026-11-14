"""Recurrent blocks over a short sequence of frame vectors."""

from torch import Tensor, nn


class LastStateLstm(nn.Module):
    """An LSTM over a sequence, oldest to newest, keeping the output at the last step. (B, K, D) -> (B, D).

    The hidden state starts at zero for every call: nothing is carried between calls.
    """

    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(width, width, num_layers=layers, batch_first=True)

    def forward(self, sequence: Tensor) -> Tensor:
        output, _ = self.lstm(sequence)
        return output[:, -1]
