"""Compact sequential chunk critic for semi-MDP IQL.

The critic consumes an encoded observation history and an action *sequence*.
Actions are kept as temporal tokens, processed by residual temporal convolutions,
and queried by the current state context through cross attention.  A gated
residual dynamics branch predicts the latent context after the complete chunk.

This module intentionally does not contain an image encoder.  The training and
evaluation scripts own an RGB encoder and save it in the same checkpoint as this
module, making the resulting critic self-contained at inference time.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    dropout: float = 0.0,
    final_layer_norm: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        layers.extend(
            [
                nn.Linear(last_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
        )
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, int(output_dim)))
    if final_layer_norm:
        layers.append(nn.LayerNorm(int(output_dim)))
    return nn.Sequential(*layers)


class ResidualTemporalConv(nn.Module):
    """Length-preserving temporal residual block for action tokens."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        tokens = self.conv(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = self.dropout(self.activation(self.norm(tokens)))
        return residual + tokens


class SequentialActionChunkEncoder(nn.Module):
    """Encode ``[B,H,A]`` chunks without flattening their time dimension."""

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_horizon: int,
        context_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_heads: int = 4,
        num_conv_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("action hidden_dim must be divisible by num_heads")
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.step_projection = nn.Sequential(
            nn.Linear(self.action_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.position_embedding = nn.Parameter(
            torch.empty(self.chunk_horizon, int(hidden_dim))
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.temporal_blocks = nn.ModuleList(
            [
                ResidualTemporalConv(int(hidden_dim), float(dropout))
                for _ in range(int(num_conv_layers))
            ]
        )
        self.context_query = nn.Sequential(
            nn.Linear(int(context_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(hidden_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_projection = make_mlp(
            2 * int(hidden_dim),
            (int(output_dim),),
            int(output_dim),
            dropout=float(dropout),
            final_layer_norm=True,
        )

    def forward(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B,H,A], got {tuple(actions.shape)}")
        batch_size, horizon, action_dim = actions.shape
        if horizon != self.chunk_horizon or action_dim != self.action_dim:
            raise ValueError(
                "action shape mismatch: expected "
                f"[B,{self.chunk_horizon},{self.action_dim}], got {tuple(actions.shape)}"
            )

        if action_mask is None:
            valid = torch.ones(
                (batch_size, horizon), dtype=torch.bool, device=actions.device
            )
        else:
            if action_mask.shape != actions.shape[:2]:
                raise ValueError(
                    f"action_mask must be [B,H], got {tuple(action_mask.shape)}"
                )
            valid = action_mask.bool()
        empty = valid.sum(dim=1) == 0
        if torch.any(empty):
            valid = valid.clone()
            valid[empty, 0] = True

        weights = valid.to(actions.dtype).unsqueeze(-1)
        tokens = self.step_projection(actions)
        tokens = (tokens + self.position_embedding[None, :, :]) * weights
        for block in self.temporal_blocks:
            tokens = block(tokens) * weights

        query = self.context_query(context).unsqueeze(1)
        attended, _ = self.cross_attention(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended = attended.squeeze(1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output_projection(torch.cat([attended, pooled], dim=-1))


class ChunkIQLDynamicsCritic(nn.Module):
    """Twin-Q chunk critic, state value, and action-conditioned latent dynamics."""

    def __init__(
        self,
        *,
        feature_dim: int,
        action_dim: int,
        chunk_horizon: int,
        latent_dim: int = 256,
        action_hidden_dim: int = 128,
        hidden_dims: Sequence[int] = (256, 256),
        num_attention_heads: int = 4,
        num_action_conv_layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.latent_dim = int(latent_dim)

        self.context_encoder = make_mlp(
            self.feature_dim,
            (self.latent_dim,),
            self.latent_dim,
            dropout=float(dropout),
            final_layer_norm=True,
        )
        self.action_encoder = SequentialActionChunkEncoder(
            action_dim=self.action_dim,
            chunk_horizon=self.chunk_horizon,
            context_dim=self.latent_dim,
            hidden_dim=int(action_hidden_dim),
            output_dim=self.latent_dim,
            num_heads=int(num_attention_heads),
            num_conv_layers=int(num_action_conv_layers),
            dropout=float(dropout),
        )
        self.state_action_fusion = make_mlp(
            2 * self.latent_dim,
            (self.latent_dim,),
            self.latent_dim,
            dropout=float(dropout),
            final_layer_norm=True,
        )
        self.transition_delta = make_mlp(
            self.latent_dim,
            hidden_dims,
            self.latent_dim,
            dropout=float(dropout),
        )
        self.transition_gate = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Sigmoid(),
        )
        self.next_context_norm = nn.LayerNorm(self.latent_dim)

        # Q sees the present state, the sequential action summary, and the
        # predicted state change. The two heads remain independent after this
        # shared representation.
        q_input_dim = 3 * self.latent_dim
        self.q1_head = make_mlp(
            q_input_dim, hidden_dims, 1, dropout=float(dropout)
        )
        self.q2_head = make_mlp(
            q_input_dim, hidden_dims, 1, dropout=float(dropout)
        )
        self.value_head = make_mlp(
            self.latent_dim, hidden_dims, 1, dropout=float(dropout)
        )

    def encode_context(self, obs_features: torch.Tensor) -> torch.Tensor:
        if obs_features.ndim != 2 or obs_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"obs_features must be [B,{self.feature_dim}], got {tuple(obs_features.shape)}"
            )
        return self.context_encoder(obs_features)

    def action_and_successor(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_repr = self.action_encoder(context, actions, action_mask)
        fused = self.state_action_fusion(torch.cat([context, action_repr], dim=-1))
        delta = self.transition_delta(fused)
        gated_delta = self.transition_gate(fused) * delta
        next_context = self.next_context_norm(context + gated_delta)
        return action_repr, gated_delta, next_context

    def q_values_from_context(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_repr, delta, _ = self.action_and_successor(
            context, actions, action_mask
        )
        q_input = torch.cat([context, action_repr, delta], dim=-1)
        return self.q1_head(q_input), self.q2_head(q_input)

    def q_min_from_context(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q1, q2 = self.q_values_from_context(context, actions, action_mask)
        return torch.minimum(q1, q2)

    def value_from_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.value_head(context)

    def forward(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context = self.encode_context(obs_features)
        action_repr, delta, next_context = self.action_and_successor(
            context, actions, action_mask
        )
        q_input = torch.cat([context, action_repr, delta], dim=-1)
        return {
            "context": context,
            "action_repr": action_repr,
            "predicted_delta": delta,
            "predicted_next_context": next_context,
            "q1": self.q1_head(q_input),
            "q2": self.q2_head(q_input),
            "v": self.value_from_context(context),
        }

