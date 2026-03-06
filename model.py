import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional


"""
    - Action State Flow (ASF), a dynamics-aware framework
    - Author: JunYoung Park, Gyuyeon Lim, and Myung-Kyu Yi
"""


class LatentEncoder(nn.Module):
    """Latent state encoder."""
    def __init__(
        self, 
        input_channels: int,
        latent_dim: int,
        hidden_dim: int
    ):
        super().__init__()

        c1_channels = hidden_dim // 2
        c2_channels = hidden_dim

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=c1_channels,
            kernel_size=5,
            padding=2,
        )
        self.bn1 = nn.BatchNorm1d(c1_channels)
        
        self.conv2 = nn.Conv1d(
            in_channels=c1_channels,
            out_channels=c2_channels,
            kernel_size=5,
            padding=2,
        )
        self.bn2 = nn.BatchNorm1d(c2_channels)
        
        self.conv3 = nn.Conv1d(
            in_channels=c2_channels,
            out_channels=latent_dim,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(latent_dim)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        s = F.relu(self.bn3(self.conv3(h)))
        s = s.transpose(1, 2)
        return s


class FlowComputer(nn.Module):
    """Latent flow feature constructor."""
    def __init__(self):
        super().__init__()

    def forward(
        self,
        s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = s.shape
        flow_raw = s[:, 1:, :] - s[:, :-1, :]
        flow_mag = torch.norm(flow_raw, dim=-1, keepdim=True)
        flow_dir = flow_raw / (flow_mag + 1e-8)

        flow_features = torch.cat(
            [flow_raw, flow_mag.expand(-1, -1, D), flow_dir], 
            dim=-1
        )
        return flow_features, flow_raw, flow_mag


class FlowEncoder(nn.Module):
    """Flow representation encoder."""
    def __init__(
        self,
        flow_dim: int,
        hidden_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.flow_embed = nn.Linear(flow_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True
        )

        self.flow_conv1 = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.flow_conv2 = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=1, 
            padding=0
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(
        self, 
        flow_features: torch.Tensor
    ) -> torch.Tensor:
        h = self.flow_embed(flow_features)
        h_att, _ = self.attention(h, h, h)
        h_att = h_att.transpose(1, 2)
        
        h = F.relu(self.bn1(self.flow_conv1(h_att)))
        h = F.relu(self.bn2(self.flow_conv2(h)))
        
        h_pool = torch.mean(h, dim=-1)
        return h_pool


class StateTransitionPredictor(nn.Module):
    """One-step latent transition predictor."""
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, 
        s_t: torch.Tensor
    ) -> torch.Tensor:
        B, Tm1, D = s_t.shape
        inp = s_t.reshape(B * Tm1, D)
        out = self.net(inp)
        return out.reshape(B, Tm1, D)


class ASFClassifier(nn.Module):
    """ASF classification model."""
    def __init__(
        self, 
        input_channels: int, 
        latent_dim: int, 
        hidden_dim: int,
        num_classes: int, 
        num_heads: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.latent_encoder = LatentEncoder(
            input_channels=input_channels, 
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        self.flow_computer = FlowComputer()

        self.flow_encoder = FlowEncoder(
            flow_dim=latent_dim * 3, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads
        )

        self.state_predictor = StateTransitionPredictor(
            latent_dim=latent_dim, 
            hidden_dim=hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.flow_prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.dynamic_gate = nn.Parameter(torch.randn(num_classes))

    def forward(
        self, 
        x: torch.Tensor, 
        return_details: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        s = self.latent_encoder(x)
        s_t = s[:, :-1, :]
        s_next = s[:, 1:, :]

        s_pred_next = self.state_predictor(s_t)

        flow_features, flow_raw, flow_mag = self.flow_computer(s)
        h = self.flow_encoder(flow_features)

        logits = self.classifier(h)

        if not return_details:
            return logits

        details = {
            "s_next": s_next,
            "s_pred_next": s_pred_next,
            "flow_mag": flow_mag,
            "h": h,
            "prototypes": self.flow_prototypes,
            "dynamic_gate": torch.sigmoid(self.dynamic_gate),
        }
        return logits, details
