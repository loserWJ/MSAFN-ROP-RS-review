"""Local capacity/dropout variants of the manuscript MSAFN; branches and fusion stay intact."""
import torch
from torch import nn

from m import MSAFN


class ConfigurableTower(nn.Module):
    def __init__(self, hidden=(16, 6), dropout=0.45):
        super().__init__()
        widths = (26, *hidden, 1)
        layers = []
        for input_dim, output_dim in zip(widths[:-2], widths[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)))
        layers.extend((nn.Linear(widths[-2], 1), nn.Sigmoid()))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class LocalTunedMSAFN(MSAFN):
    def __init__(self, *, tower_hidden=(16, 6), tower_dropout=0.45,
                 transformer_dropout=0.7, csan_depth_layers=1, attention_width=16):
        if len(tower_hidden) not in (2, 3) or min(tower_hidden) < 1:
            raise ValueError("tower_hidden must contain two or three positive widths")
        if not 0 <= tower_dropout < 1 or not 0 <= transformer_dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if csan_depth_layers not in (1, 2):
            raise ValueError("CSAN depth experiment supports one or two BiLSTM layers")
        super().__init__(current_input_mode="separate", transformer_dropout=transformer_dropout,
                         attention_width=attention_width)
        if csan_depth_layers > 1:
            self.mwdt.depth_lstm = nn.LSTM(
                8, 32, num_layers=csan_depth_layers, batch_first=True,
                bidirectional=True, dropout=0.1,
            )
        self.tower_a = ConfigurableTower(tower_hidden, tower_dropout)
        self.tower_b = ConfigurableTower(tower_hidden, tower_dropout)
        self.config = dict(tower_hidden=tuple(tower_hidden), tower_dropout=tower_dropout,
                           transformer_dropout=transformer_dropout,
                           csan_depth_layers=csan_depth_layers,
                           attention_width=attention_width)
