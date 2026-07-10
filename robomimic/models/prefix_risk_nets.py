"""Networks shared by prefix-risk training and policy regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPrefixRisk(nn.Module):
    """Decompose rollout failure risk into prefix state risk and action risk."""

    def __init__(
        self,
        feature_dim: int,
        prediction_horizon: int,
        action_dim: int,
        hidden_dim: int,
        action_hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.obs_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.prefix_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(prediction_horizon * action_dim, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.action_residual = nn.Sequential(
            nn.Linear(hidden_dim + action_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_prefix(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.obs_projection(features)
        context, _ = self.prefix_encoder(projected)
        return context

    def action_delta(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return incremental action risk in failure log-odds units.

        ``context`` has shape ``[B, T, H]`` and ``actions`` has shape
        ``[B, T, prediction_horizon, action_dim]``.
        """
        encoded_action = self.action_encoder(actions.flatten(start_dim=2))
        return self.action_residual(
            torch.cat([context, encoded_action], dim=-1)
        ).squeeze(-1)

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        context = self.encode_prefix(features)
        state_logit = self.state_head(context).squeeze(-1)
        delta = self.action_delta(context, actions)
        action_logit = state_logit.detach() + delta
        return {
            "context": context,
            "state_logit": state_logit,
            "action_delta": delta,
            "action_logit": action_logit,
        }


class CausalTemporalConvBlock(nn.Module):
    """Small residual causal temporal block for prefix features."""

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv1d(dim, dim, kernel_size=self.kernel_size)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]. Pad only on the left so context_t cannot see future
        # observation features.
        y = x.transpose(1, 2)
        y = F.pad(y, (self.kernel_size - 1, 0))
        y = self.conv(y).transpose(1, 2)
        y = self.dropout(self.activation(self.norm(y)))
        return x + y


class TemporalConvBlock(nn.Module):
    """Small residual temporal block for a full candidate action chunk.

    The action chunk is already known to the scorer, so this block is
    non-causal inside the 16-step candidate. That lets the scorer inspect
    approach-contact-lift structure across the whole proposed chunk.
    """

    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, L, C]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(self.activation(self.norm(y)))
        return x + y


class ActionChunkCrossAttentionEncoder(nn.Module):
    """Encode a sequential action chunk and query it with current context."""

    def __init__(
        self,
        *,
        action_dim: int,
        prediction_horizon: int,
        context_dim: int,
        action_hidden_dim: int,
        num_heads: int,
        num_conv_layers: int,
        dropout: float,
    ):
        super().__init__()
        if action_hidden_dim % num_heads != 0:
            raise ValueError(
                "action_hidden_dim must be divisible by num_heads for attention"
            )
        self.prediction_horizon = int(prediction_horizon)
        self.step_projection = nn.Sequential(
            nn.Linear(action_dim, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(self.prediction_horizon, action_hidden_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.temporal_encoder = nn.Sequential(
            *[
                TemporalConvBlock(
                    action_hidden_dim,
                    kernel_size=3,
                    dropout=dropout,
                )
                for _ in range(max(int(num_conv_layers), 0))
            ]
        )
        self.context_query = nn.Sequential(
            nn.Linear(context_dim, action_hidden_dim),
            nn.LayerNorm(action_hidden_dim),
            nn.SiLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=action_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(action_hidden_dim * 2)

    def forward(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        # context: [B, T, H], actions: [B, T, L, A]
        batch_size, seq_len, horizon, action_dim = actions.shape
        if horizon != self.prediction_horizon:
            raise ValueError(
                f"expected action horizon={self.prediction_horizon}, got {horizon}"
            )
        flat_actions = actions.reshape(batch_size * seq_len, horizon, action_dim)
        tokens = self.step_projection(flat_actions)
        tokens = tokens + self.position_embedding[None, :, :]
        tokens = self.temporal_encoder(tokens)

        flat_context = context.reshape(batch_size * seq_len, context.shape[-1])
        query = self.context_query(flat_context).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        attended = attended.squeeze(1)
        pooled = tokens.mean(dim=1)
        encoded = self.output_norm(torch.cat([attended, pooled], dim=-1))
        return encoded.reshape(batch_size, seq_len, -1)


class CausalPrefixRiskV2(nn.Module):
    """Stronger causal prefix outcome model with sequential action scoring.

    Compared with :class:`CausalPrefixRisk`, V2 keeps the same public outputs
    and checkpoint semantics, but improves the action branch:

    * prefix features are encoded by causal temporal conv blocks plus GRU;
    * the 16-step action chunk is encoded as a sequence, not flattened first;
    * current context queries action tokens with cross-attention;
    * deeper MLP heads predict state logit and action-conditioned residual.
    """

    def __init__(
        self,
        feature_dim: int,
        prediction_horizon: int,
        action_dim: int,
        hidden_dim: int,
        action_hidden_dim: int,
        dropout: float,
        *,
        action_num_heads: int = 4,
        action_conv_layers: int = 2,
        prefix_conv_layers: int = 1,
    ):
        super().__init__()
        self.obs_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.prefix_temporal_encoder = nn.Sequential(
            *[
                CausalTemporalConvBlock(hidden_dim, kernel_size=3, dropout=dropout)
                for _ in range(max(int(prefix_conv_layers), 0))
            ]
        )
        self.prefix_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.action_encoder = ActionChunkCrossAttentionEncoder(
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            context_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
            num_heads=action_num_heads,
            num_conv_layers=action_conv_layers,
            dropout=dropout,
        )
        self.action_residual = nn.Sequential(
            nn.Linear(hidden_dim + 2 * action_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_prefix(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.obs_projection(features)
        projected = self.prefix_temporal_encoder(projected)
        context, _ = self.prefix_encoder(projected)
        return self.context_norm(context)

    def action_delta(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return incremental outcome log-odds for a candidate action chunk."""
        encoded_action = self.action_encoder(context, actions)
        return self.action_residual(
            torch.cat([context, encoded_action], dim=-1)
        ).squeeze(-1)

    def forward(self, features: torch.Tensor, actions: torch.Tensor):
        context = self.encode_prefix(features)
        state_logit = self.state_head(context).squeeze(-1)
        delta = self.action_delta(context, actions)
        action_logit = state_logit.detach() + delta
        return {
            "context": context,
            "state_logit": state_logit,
            "action_delta": delta,
            "action_logit": action_logit,
        }


def make_causal_prefix_model(
    *,
    model_arch: str,
    feature_dim: int,
    prediction_horizon: int,
    action_dim: int,
    hidden_dim: int,
    action_hidden_dim: int,
    dropout: float,
    action_num_heads: int = 4,
    action_conv_layers: int = 2,
    prefix_conv_layers: int = 1,
) -> nn.Module:
    """Factory used by training and rollout scoring.

    ``model_arch='v1'`` preserves old checkpoints. ``model_arch='v2'`` enables
    the stronger sequential-action cross-attention scorer.
    """
    if model_arch in ("v1", "flat", "causal_prefix_risk"):
        return CausalPrefixRisk(
            feature_dim=feature_dim,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
            dropout=dropout,
        )
    if model_arch in ("v2", "cross_attn_v2", "causal_prefix_risk_v2"):
        return CausalPrefixRiskV2(
            feature_dim=feature_dim,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
            dropout=dropout,
            action_num_heads=action_num_heads,
            action_conv_layers=action_conv_layers,
            prefix_conv_layers=prefix_conv_layers,
        )
    raise ValueError(f"unknown model_arch={model_arch}")
