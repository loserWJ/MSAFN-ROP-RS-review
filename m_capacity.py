"""Local capacity/information-preservation variants of the manuscript MSAFN."""
import torch
from torch import nn
from m import DNN, FeatureEnhancer, FeatureTokens, Attention, ResidualFusion
from m_local_tuned import ConfigurableTower
from m_structural import StructuralMSAFN


class ResidualEnhancer(FeatureEnhancer):
    def __init__(self, width=16, features=14):
        super().__init__(width)
        if features != 14:
            self.current_tokens = FeatureTokens(features, width)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))

    def forward(self, current, context):
        a, b = self.current_tokens(current), self.context_tokens(context)
        a = self.norms[0](a + self.current_self(a, a))
        b = self.norms[1](b + self.context_self(b, b))
        b = self.norms[2](b + self.update_context(b, a))
        a = self.norms[3](a + self.update_current(a, b))
        return a + self.current_dnn(a), b + self.context_dnn(b)


class ResidualCross(Attention):
    def __init__(self, width):
        super().__init__(width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, query, context):
        return self.norm(query + super().forward(query, context))


class CapacityMSAFN(StructuralMSAFN):
    def __init__(self, *, enhancer_residual=False, context_residual=False,
                 wider=False, csan_balance=False):
        # Baseline initialization matches the prior controlled structural study.
        super().__init__()
        current, width, latent = (32, 32, 24) if wider else (14, 16, 12)
        if wider:
            self.dnn1_blast = DNN([5, 64, 32, 16], batch_norm=True)
            self.dnn2_blast = DNN([5, 64, 32, 16], batch_norm=True)
            self.enhancer = FeatureEnhancer(width)
            self.enhancer.current_tokens = FeatureTokens(current, width)
            self.final_cross = Attention(width, width, width)
            self.fusion = nn.Sequential(ResidualFusion(current*width,128),
                                        ResidualFusion(128,64), ResidualFusion(64,latent))
            self.tower_a = self.make_tower(current+latent)
            self.tower_b = self.make_tower(current+latent)
        if enhancer_residual:
            self.enhancer = ResidualEnhancer(width, current)
            self.final_cross = ResidualCross(width)
        if context_residual:
            self.context_skip = nn.Linear(64, latent, bias=False)
            self.context_gate = nn.Parameter(torch.tensor(-2.0))
        if csan_balance:
            self.mwdt = nn.Sequential(self.mwdt, nn.BatchNorm1d(32))
        self.config = dict(enhancer_residual=enhancer_residual,
                           context_residual=context_residual, wider=wider,
                           csan_balance=csan_balance)

    @staticmethod
    def make_tower(input_width):
        return nn.Sequential(nn.Linear(input_width,64),nn.SiLU(),nn.Dropout(.15),
                             nn.Linear(64,32),nn.SiLU(),nn.Dropout(.15),
                             nn.Linear(32,1),nn.Sigmoid())

    def forward_features(self, *inputs):
        features = super().forward_features(*inputs)
        if self.config['context_residual']:
            features['latent'] = features['latent'] + self.context_gate.sigmoid()*self.context_skip(features['context'])
            features['tower_input'] = torch.cat((features['latent'],features['current']),dim=-1)
        return features
