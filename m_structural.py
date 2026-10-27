"""MSAFN local structural variants retaining all four branches and both tasks.

These are optimization variants, not the unchanged manuscript hyperparameters.
"""
import math

import torch
from torch import nn

from m import Attention, DNN
from m_local_tuned import LocalTunedMSAFN


class QueryResidualAttention(Attention):
    """Keep the query representation alongside the attention-weighted values."""
    def __init__(self, query_dim, context_dim, latent_dim=64):
        super().__init__(query_dim, context_dim, latent_dim)
        self.skip = nn.Identity() if query_dim == latent_dim else nn.Linear(query_dim, latent_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, query, context):
        return super().forward(query, context) + self.residual_scale * self.skip(query)


class CurrentResidualEncoder(nn.Module):
    """Deeper current-cycle encoding with a dimension-aligned linear shortcut."""
    def __init__(self, output_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(5, 48), nn.BatchNorm1d(48), nn.SiLU(),
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.SiLU(),
            nn.Linear(32, output_dim),
        )
        self.skip = nn.Linear(5, output_dim, bias=False)

    def forward(self, x):
        return self.main(x) + self.skip(x)


class TaskResidualBlock(nn.Module):
    def __init__(self, width=48, dropout=.1):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(width, width))

    def forward(self, x):
        return x + .5 * self.main(x)


class ResidualTaskTower(nn.Module):
    def __init__(self, width=48, depth=2, dropout=.1):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(26, width), nn.SiLU())
        self.blocks = nn.Sequential(*(TaskResidualBlock(width, dropout) for _ in range(depth)))
        self.output = nn.Sequential(nn.Linear(width, 1), nn.Sigmoid())

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


class StructuralMSAFN(LocalTunedMSAFN):
    def __init__(self, *, attention_residual=False, current_residual=False,
                 task_residual=False, task_width=48, task_depth=2):
        # Identical baseline initialization before any modules are replaced.
        super().__init__(tower_hidden=(32, 12))
        if attention_residual:
            replacements = []
            for name, module in self.named_modules():
                if isinstance(module, Attention):
                    replacements.append((name, module))
            for name, module in replacements:
                new = QueryResidualAttention(module.q.in_features, module.k.in_features,
                                             module.q.out_features)
                new.load_state_dict(module.state_dict(), strict=False)
                parent_name, _, child = name.rpartition('.')
                parent = self.get_submodule(parent_name) if parent_name else self
                setattr(parent, child, new)
        if current_residual:
            self.dnn1_blast = CurrentResidualEncoder(6)
            self.dnn2_blast = CurrentResidualEncoder(8)
        if task_residual:
            self.tower_a = ResidualTaskTower(task_width, task_depth)
            self.tower_b = ResidualTaskTower(task_width, task_depth)
        self.config = dict(attention_residual=attention_residual,
                           current_residual=current_residual,
                           task_residual=task_residual,
                           task_width=task_width, task_depth=task_depth)


if __name__ == '__main__':
    model = StructuralMSAFN(attention_residual=True, current_residual=True, task_residual=True)
    args = (torch.rand(3, 7, 75, 8), torch.rand(3, 30, 6), torch.rand(3, 5), torch.rand(3, 5))
    out = model(*args)
    sum(x.mean() for x in out).backward()
    print([tuple(x.shape) for x in out])
