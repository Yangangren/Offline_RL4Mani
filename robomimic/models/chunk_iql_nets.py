"""Compact sequential chunk critics for semi-MDP IQL.

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


class CausalTemporalStateTrunk(nn.Module):
    """Turn chronological frame latents into causal temporal state tokens.

    The final token can only attend to the supplied history (never a future
    frame). Keeping this module separate from the image encoder makes the
    temporal representation an explicit, checkpointed part of the critic.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        max_history: int,
        num_layers: int = 2,
        num_heads: int = 3,
        feedforward_dim: int = 600,
        dropout: float = 0.0,
    ):
        super().__init__()
        state_dim = int(state_dim)
        num_heads = int(num_heads)
        if state_dim <= 0 or int(max_history) <= 0:
            raise ValueError("state_dim and max_history must be positive")
        if int(num_layers) <= 0 or int(feedforward_dim) <= 0:
            raise ValueError("temporal layer and feedforward counts must be positive")
        if state_dim % num_heads != 0:
            raise ValueError(
                f"temporal state_dim={state_dim} must be divisible by "
                f"num_heads={num_heads}"
            )
        self.state_dim = state_dim
        self.max_history = int(max_history)
        self.position_embedding = nn.Parameter(
            torch.empty(self.max_history, self.state_dim)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.state_dim,
            nhead=num_heads,
            dim_feedforward=int(feedforward_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(self.state_dim)

    def forward(self, frame_latents: torch.Tensor) -> torch.Tensor:
        if frame_latents.ndim != 3:
            raise ValueError(
                "frame_latents must be [B,T,D], got "
                f"{tuple(frame_latents.shape)}"
            )
        _, history, state_dim = frame_latents.shape
        if not 1 <= int(history) <= self.max_history:
            raise ValueError(
                f"history must be in [1,{self.max_history}], got {history}"
            )
        if int(state_dim) != self.state_dim:
            raise ValueError(
                f"frame latent dim={state_dim} does not match {self.state_dim}"
            )
        tokens = frame_latents + self.position_embedding[None, :history]
        causal_mask = torch.triu(
            torch.ones(
                (history, history),
                dtype=torch.bool,
                device=frame_latents.device,
            ),
            diagonal=1,
        )
        return self.output_norm(self.transformer(tokens, mask=causal_mask))


class ResidualActionLatentRollout(nn.Module):
    """Recurrently predict future frame latents at fixed action offsets.

    A GRU carries the action-prefix state. The predicted frame latent is
    updated residually at every action, so predictions at offsets 2, 4, 6,
    and 8 are genuinely prefix-conditioned rather than four independent
    regressions over a flattened chunk.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_horizon: int,
        state_dim: int,
        prediction_offsets: Sequence[int],
        hidden_dims: Sequence[int] = (300, 300),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_horizon = int(chunk_horizon)
        self.state_dim = int(state_dim)
        self.prediction_offsets = tuple(int(value) for value in prediction_offsets)
        if self.action_dim <= 0 or self.chunk_horizon <= 0 or self.state_dim <= 0:
            raise ValueError("action, chunk, and state dimensions must be positive")
        if (
            not self.prediction_offsets
            or tuple(sorted(set(self.prediction_offsets))) != self.prediction_offsets
            or self.prediction_offsets[0] < 1
            or self.prediction_offsets[-1] > self.chunk_horizon
        ):
            raise ValueError(
                "prediction_offsets must be sorted, unique, positive, and no "
                f"larger than chunk_horizon={self.chunk_horizon}; got "
                f"{self.prediction_offsets}"
            )
        self.action_projection = nn.Sequential(
            nn.Linear(self.action_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),
            nn.SiLU(),
        )
        self.action_position_embedding = nn.Parameter(
            torch.empty(self.chunk_horizon, self.state_dim)
        )
        nn.init.normal_(self.action_position_embedding, mean=0.0, std=0.02)
        self.prefix_cell = nn.GRUCell(self.state_dim, self.state_dim)
        self.offset_embedding = nn.Parameter(
            torch.empty(len(self.prediction_offsets), self.state_dim)
        )
        nn.init.normal_(self.offset_embedding, mean=0.0, std=0.02)
        self.condition_norm = nn.LayerNorm(self.state_dim)
        self.film = nn.Linear(2 * self.state_dim, 3 * self.state_dim)
        self.residual_block = make_mlp(
            self.state_dim,
            tuple(int(value) for value in hidden_dims),
            self.state_dim,
            dropout=float(dropout),
        )
        self.delta_output = nn.Linear(self.state_dim, self.state_dim)
        # WCM-style near-identity initialization: the gated residual begins
        # mostly closed and the final latent delta begins near zero.
        with torch.no_grad():
            self.film.bias[2 * self.state_dim :].fill_(-2.0)
        nn.init.normal_(self.delta_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_output.bias)

    def forward(
        self,
        temporal_state: torch.Tensor,
        current_frame_latent: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temporal_state.ndim != 2 or current_frame_latent.ndim != 2:
            raise ValueError("temporal and current-frame states must be [B,D]")
        if temporal_state.shape != current_frame_latent.shape:
            raise ValueError(
                "temporal and current-frame states must have identical shapes"
            )
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.chunk_horizon,
            self.action_dim,
        ):
            raise ValueError(
                "dynamics actions must be "
                f"[B,{self.chunk_horizon},{self.action_dim}], got "
                f"{tuple(actions.shape)}"
            )
        if int(temporal_state.shape[-1]) != self.state_dim:
            raise ValueError(
                f"state dim={temporal_state.shape[-1]} does not match "
                f"{self.state_dim}"
            )
        if action_mask is None:
            valid = actions.new_ones(actions.shape[:2])
        else:
            if action_mask.shape != actions.shape[:2]:
                raise ValueError(
                    f"action_mask must be [B,H], got {tuple(action_mask.shape)}"
                )
            valid = action_mask.to(dtype=actions.dtype)

        hidden = temporal_state
        predictions: list[torch.Tensor] = []
        offset_to_index = {
            offset: index
            for index, offset in enumerate(self.prediction_offsets)
        }
        for step in range(self.chunk_horizon):
            action_token = (
                self.action_projection(actions[:, step])
                + self.action_position_embedding[step]
            )
            candidate_hidden = self.prefix_cell(action_token, hidden)
            step_valid = valid[:, step : step + 1]
            hidden = step_valid * candidate_hidden + (1.0 - step_valid) * hidden
            offset_index = offset_to_index.get(step + 1)
            if offset_index is not None:
                film_input = torch.cat(
                    (
                        hidden,
                        self.offset_embedding[offset_index]
                        .unsqueeze(0)
                        .expand(hidden.shape[0], -1),
                    ),
                    dim=-1,
                )
                shift, scale, gate = self.film(film_input).chunk(3, dim=-1)
                conditioned = (
                    self.condition_norm(temporal_state)
                    * (1.0 + scale)
                    + shift
                )
                world_hidden = temporal_state + torch.sigmoid(gate) * (
                    self.residual_block(conditioned)
                )
                predictions.append(
                    current_frame_latent + self.delta_output(world_hidden)
                )
        return torch.stack(predictions, dim=1)


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

