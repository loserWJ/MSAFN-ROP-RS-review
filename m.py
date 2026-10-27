"""MSAFN reconstructed from the supplied manuscript, Figures 4 and 5.

Inputs: previous MWD [B,7,75,8] (or paper layout [B,7,8,75]),
construction [B,30,6], rock [B,5], active controls [B,5].
Outputs: (normalized ROP [B,1], normalized rotation speed RS [B,1]).

The manuscript is internally inconsistent about the current branches:
``separate`` follows Figure 5; ``joint10`` follows the stated 10-input layers.
Unspecified implementation details are documented in the accompanying README.
Importing this module never loads data or starts training.
"""
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Single-head scaled dot-product attention with explicit Q/K/V mappings."""

    def __init__(self, query_dim, context_dim, latent_dim=64):
        super().__init__()
        self.q = nn.Linear(query_dim, latent_dim)
        self.k = nn.Linear(context_dim, latent_dim)
        self.v = nn.Linear(context_dim, latent_dim)
        self.scale = math.sqrt(latent_dim)

    def forward(self, query, context):
        weights = torch.softmax(
            self.q(query) @ self.k(context).transpose(-2, -1) / self.scale,
            dim=-1,
        )
        return weights @ self.v(context)


class HoleChannelAttention(nn.Module):
    """SE-style gating on the SEVEN HOLES, not the eight MWD attributes.

    Pool across depth and attributes; the non-shared 7→7→7 gate retains
    fixed hole identities. Internal gate width is an implementation choice.
    """

    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(7, 7), nn.ReLU(), nn.Linear(7, 7), nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.gate(x.mean(dim=(2, 3)))
        return x * weights[:, :, None, None]


class CSAN(nn.Module):
    """Figure 4 and CSAN equations (2)–(18); no GAT or added position layer."""

    def __init__(self):
        super().__init__()
        # Total bidirectional hidden width is 64, NOT 64 per direction.
        self.depth_lstm = nn.LSTM(8, 32, batch_first=True, bidirectional=True)
        self.depth_self = Attention(64, 64)
        self.hole_attention = HoleChannelAttention()
        self.lateral_self = Attention(8, 8)
        self.lateral_to_depth = Attention(64, 64)
        self.depth_to_lateral = Attention(64, 64)
        # The text specifies channel alignment, but no kernel size: use 1.
        self.depth_align = nn.Conv1d(64, 32, kernel_size=1)
        self.lateral_align = nn.Conv1d(64, 32, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("MWD must be [B,7,75,8] or [B,7,8,75].")
        if tuple(x.shape[1:]) == (7, 8, 75):
            x = x.permute(0, 1, 3, 2)
        if tuple(x.shape[1:]) != (7, 75, 8):
            raise ValueError("CSAN requires exactly 7 holes, 75 depths, 8 attributes.")
        b, h, d, c = x.shape
        depth, _ = self.depth_lstm(x.reshape(b * h, d, c))
        depth = self.depth_self(depth, depth).reshape(b, h, d, 64)
        lateral = self.hole_attention(x).permute(0, 2, 1, 3)
        lateral = lateral.reshape(b * d, h, c)
        lateral = self.lateral_self(lateral, lateral).reshape(b, d, h, 64)

        # Align BOTH sample and spatial indexing before cross-attention.
        enhanced_depth = self.lateral_to_depth(
            depth.reshape(b * h, d, 64),
            lateral.permute(0, 2, 1, 3).reshape(b * h, d, 64),
        )
        enhanced_lateral = self.depth_to_lateral(
            lateral.reshape(b * d, h, 64),
            depth.permute(0, 2, 1, 3).reshape(b * d, h, 64),
        )
        zd = self.depth_align(enhanced_depth.transpose(1, 2))
        zd = zd.transpose(1, 2).reshape(b, h, d, 32)
        zl = self.lateral_align(enhanced_lateral.transpose(1, 2))
        zl = zl.transpose(1, 2).reshape(b, d, h, 32).permute(0, 2, 1, 3)
        # Equation (18) specifies SUM, followed by Agg; choose mean for Agg.
        return (zd + zl).mean(dim=(1, 2))


class PositionalEncoding(nn.Module):
    """Construction branch only; position is axis 1 for batch-first input."""

    def __init__(self, d_model=4, max_len=30):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype)


class DNN(nn.Module):
    def __init__(self, widths, batch_norm=False):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(widths) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(b))
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConstructionEncoder(nn.Module):
    def __init__(self, feedforward_dim=512, dropout=0.7):
        super().__init__()
        # FFN width/dropout inherit the old script; manuscript does not set them.
        self.embedding = nn.Linear(6, 4)
        self.position = PositionalEncoding()
        layer = nn.TransformerEncoderLayer(
            d_model=4, nhead=4, dim_feedforward=feedforward_dim,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        # Initialize cloned layers independently.
        for block in self.transformer.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        self.transformer_projection = nn.Linear(30 * 4, 16)
        self.dnn = DNN([180, 128, 64, 32], batch_norm=True)
        self.dnn_projection = nn.Linear(32, 16)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (30, 6):
            raise ValueError("Construction must be [B,30,6]; transpose legacy [B,6,30].")
        seq = self.position(self.embedding(x) * math.sqrt(4))
        a = self.transformer_projection(self.transformer(seq).flatten(1))
        b = self.dnn_projection(self.dnn(x.flatten(1)))
        return a, b


class FeatureTokens(nn.Module):
    """One scalar feature per token with learned feature-specific embedding.

    Tokenization is not specified by the manuscript. Using multiple tokens
    avoids the degenerate softmax=1 of attention on a single pooled vector.
    """

    def __init__(self, features, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features, width))
        self.bias = nn.Parameter(torch.zeros(features, width))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias


class FeatureEnhancer(nn.Module):
    """Fig. 5b: self-attention, current-guided context, context-guided current."""

    def __init__(self, width=16):
        super().__init__()
        self.current_tokens = FeatureTokens(14, width)
        self.context_tokens = FeatureTokens(64, width)
        self.current_self = Attention(width, width, width)
        self.context_self = Attention(width, width, width)
        self.update_context = Attention(width, width, width)
        self.update_current = Attention(width, width, width)
        self.current_dnn = DNN([width, width, width])
        self.context_dnn = DNN([width, width, width])

    def forward(self, current, context):
        a = self.current_tokens(current)
        b = self.context_tokens(context)
        a = self.current_self(a, a)
        b = self.context_self(b, b)
        # A=current is the K,V source first; B=context is Q.
        b = self.update_context(b, a)
        a = self.update_current(a, b)
        return self.current_dnn(a), self.context_dnn(b)


class ResidualFusion(nn.Module):
    """Fig. 5c: FC, ReLU, LayerNorm, plus a dimension-aligned input skip."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = (nn.Identity() if input_dim == output_dim
                     else nn.Linear(input_dim, output_dim, bias=False))

    def forward(self, x):
        return self.norm(F.relu(self.fc(x))) + self.skip(x)


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(26, 16), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(16, 6), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(6, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class MultiModalModel(nn.Module):
    def __init__(self, time_series_input_dim=6, node_feature_dim=1, *,
                 current_input_mode="separate", attention_width=16,
                 transformer_feedforward=512, transformer_dropout=0.7):
        super().__init__()
        if time_series_input_dim != 6 or node_feature_dim != 1:
            raise ValueError("Legacy compatibility arguments must be 6 and 1.")
        if current_input_mode not in ("separate", "joint10"):
            raise ValueError("current_input_mode must be 'separate' or 'joint10'.")
        if not isinstance(attention_width, int) or attention_width < 1:
            raise ValueError("attention_width must be a positive integer.")
        self.config = dict(
            current_input_mode=current_input_mode, attention_width=attention_width,
            transformer_feedforward=transformer_feedforward,
            transformer_dropout=transformer_dropout,
        )
        self.current_input_mode = current_input_mode
        self.mwdt = CSAN()
        self.construction = ConstructionEncoder(
            transformer_feedforward, transformer_dropout)
        input_width = 5 if current_input_mode == "separate" else 10
        self.dnn1_blast = DNN([input_width, 15, 10, 6], batch_norm=True)
        self.dnn2_blast = DNN([input_width, 20, 12, 8], batch_norm=True)
        self.context_fusion = ResidualFusion(64, 64)
        self.enhancer = FeatureEnhancer(attention_width)
        self.final_cross = Attention(attention_width, attention_width, attention_width)
        self.fusion = nn.Sequential(
            ResidualFusion(14 * attention_width, 64),
            ResidualFusion(64, 32),
            ResidualFusion(32, 12),
        )
        self.tower_a = Tower()
        self.tower_b = Tower()

    def forward_features(self, mwdt, const, roc, mwd):
        if roc.ndim != 2 or roc.shape[1] != 5:
            raise ValueError("Rock input must contain 5 features.")
        if mwd.ndim != 2 or mwd.shape[1] != 5:
            raise ValueError("Active input must contain 5 controls; no current ROP/RS/MSE.")
        if len({mwdt.shape[0], const.shape[0], roc.shape[0], mwd.shape[0]}) != 1:
            raise ValueError("All modalities must have the same batch size.")
        if self.current_input_mode == "joint10":
            x1 = x2 = torch.cat((roc, mwd), dim=-1)
        else:
            x1, x2 = roc, mwd
        current = torch.cat((self.dnn1_blast(x1), self.dnn2_blast(x2)), dim=-1)
        mwd_features = self.mwdt(mwdt)
        trans, con_dnn = self.construction(const)
        context = self.context_fusion(torch.cat((mwd_features, trans, con_dnn), dim=-1))
        enhanced_current, enhanced_context = self.enhancer(current, context)
        fused = self.final_cross(enhanced_current, enhanced_context)
        latent = self.fusion(fused.flatten(1))
        tower_input = torch.cat((latent, current), dim=-1)
        return dict(csan=mwd_features, transformer=trans, construction_dnn=con_dnn,
                    current=current, context=context, latent=latent,
                    tower_input=tower_input)

    def forward(self, mwdt, const, roc, mwd):
        features = self.forward_features(mwdt, const, roc, mwd)["tower_input"]
        return self.tower_a(features), self.tower_b(features)


MSAFN = MultiModalModel


class CompositeRobustLoss(nn.Module):
    """0.424 Huber(delta=1.5) + 0.357 quantile(tau=.75) + 0.219 MAE.

    'formula' follows the printed equation: underprediction is penalized more.
    'overprediction' follows the contradictory prose, reversing the residual.
    The within-task composite is unchanged. Task losses use a normalized
    weighted mean, defaulting to equal ROP and RS weights.
    """

    def __init__(self, weights=(0.424, 0.357, 0.219), delta=1.5, tau=0.75,
                 quantile_mode="formula", task_weights=(1.0, 1.0)):
        super().__init__()
        if len(weights) != 3 or any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Three finite nonnegative loss weights are required.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("Loss weights must sum to one.")
        if not math.isfinite(delta) or delta <= 0 or not 0 < tau < 1:
            raise ValueError("Require delta > 0 and 0 < tau < 1.")
        if quantile_mode not in ("formula", "overprediction"):
            raise ValueError("Unknown quantile mode.")
        if (len(task_weights) != 2 or
                any(not math.isfinite(w) or w < 0 for w in task_weights) or
                sum(task_weights) <= 0):
            raise ValueError("Two finite nonnegative task weights with positive sum are required.")
        self.weights, self.delta, self.tau = weights, delta, tau
        self.quantile_mode = quantile_mode
        total = sum(task_weights)
        self.task_weights = tuple(w / total for w in task_weights)

    def per_task(self, prediction, target):
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
            raise ValueError("Prediction and target must both have shape [B,2] (ROP,RS).")
        error = target - prediction
        if self.quantile_mode == "overprediction":
            error = -error
        quantile = torch.maximum(self.tau * error, (self.tau - 1) * error)
        huber = F.huber_loss(prediction, target, reduction="none", delta=self.delta)
        mae = (prediction - target).abs()
        return (self.weights[0] * huber + self.weights[1] * quantile
                + self.weights[2] * mae).mean(dim=0)

    def forward(self, prediction, target):
        return (self.per_task(prediction, target) *
                prediction.new_tensor(self.task_weights)).sum()


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MSAFN().eval()
    inputs = (torch.rand(2, 7, 75, 8), torch.rand(2, 30, 6),
              torch.rand(2, 5), torch.rand(2, 5))
    with torch.no_grad():
        print({k: tuple(v.shape) for k, v in model.forward_features(*inputs).items()})
        print("ROP, RS:", [tuple(y.shape) for y in model(*inputs)])
    warnings.warn("Structure smoke test only. See the fidelity report for manuscript ambiguities.")
