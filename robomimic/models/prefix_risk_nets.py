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


class CausalSuccessorCriticV3(nn.Module):
    """Action-conditioned latent dynamics plus a success-value critic.

    The state branch estimates V(h_t). The transition branch predicts the
    latent context reached after an action chunk, and the same value head
    evaluates that successor. Thus action_delta is Q(h_t, a) - V(h_t)
    in success-logit space, while keeping the existing extraction interface.
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
        transition_input_dim = hidden_dim + 2 * action_hidden_dim
        self.transition_model = nn.Sequential(
            nn.Linear(transition_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.transition_gate = nn.Sequential(
            nn.Linear(transition_input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.successor_norm = nn.LayerNorm(hidden_dim)

    def encode_prefix(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.obs_projection(features)
        projected = self.prefix_temporal_encoder(projected)
        context, _ = self.prefix_encoder(projected)
        return self.context_norm(context)

    def predict_next_context(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_action = self.action_encoder(context, actions, action_mask)
        transition_input = torch.cat([context, encoded_action], dim=-1)
        residual = self.transition_model(transition_input)
        gate = self.transition_gate(transition_input)
        return self.successor_norm(context + gate * residual)

    def action_value_logit(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted_next_context = self.predict_next_context(
            context,
            actions,
            action_mask,
        )
        return self.state_head(predicted_next_context).squeeze(-1)

    def action_delta(
        self,
        context: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current_logit = self.state_head(context).squeeze(-1)
        action_logit = self.action_value_logit(context, actions, action_mask)
        return action_logit - current_logit.detach()

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ):
        context = self.encode_prefix(features)
        state_logit = self.state_head(context).squeeze(-1)
        predicted_next_context = self.predict_next_context(
            context,
            actions,
            action_mask,
        )
        action_logit = self.state_head(predicted_next_context).squeeze(-1)
        delta = action_logit - state_logit.detach()
        return {
            "context": context,
            "state_logit": state_logit,
            "predicted_next_context": predicted_next_context,
            "action_delta": delta,
            "action_logit": action_logit,
        }


class RGBSuccessorCriticV4(CausalSuccessorCriticV3):
    """Self-contained RGB-to-value successor critic.

    The model owns the visual observation encoder used to produce every
    boundary feature. The inherited ``forward`` preserves efficient training
    from a feature cache while ``forward_rgb`` is the deployment interface. Saving
    this module's state dict therefore saves the image encoder together with
    the temporal value and action-conditioned dynamics heads.
    """

    def __init__(
        self,
        *,
        rgb_encoder: nn.Module,
        observation_shapes: dict[str, tuple[int, ...]],
        observation_horizon: int,
        feature_dim: int,
        prediction_horizon: int,
        action_dim: int,
        hidden_dim: int,
        action_hidden_dim: int,
        dropout: float,
        feature_mean,
        feature_std,
        action_mean,
        action_std,
        action_num_heads: int = 4,
        action_conv_layers: int = 2,
        prefix_conv_layers: int = 1,
    ):
        super().__init__(
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
        if observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive")
        self.rgb_encoder = rgb_encoder
        self.observation_shapes = {
            key: tuple(shape) for key, shape in observation_shapes.items()
        }
        self.observation_horizon = int(observation_horizon)
        encoder_dim = int(self.rgb_encoder.output_shape()[0])
        if encoder_dim * self.observation_horizon != int(feature_dim):
            raise ValueError(
                "RGB encoder output does not match critic feature_dim: "
                f"encoder_dim={encoder_dim}, observation_horizon="
                f"{self.observation_horizon}, feature_dim={feature_dim}"
            )

        feature_mean = torch.as_tensor(feature_mean, dtype=torch.float32).reshape(-1)
        feature_std = torch.as_tensor(feature_std, dtype=torch.float32).reshape(-1)
        action_mean = torch.as_tensor(action_mean, dtype=torch.float32).reshape(-1)
        action_std = torch.as_tensor(action_std, dtype=torch.float32).reshape(-1)
        if feature_mean.numel() != int(feature_dim):
            raise ValueError("feature_mean does not match feature_dim")
        if feature_std.numel() != int(feature_dim):
            raise ValueError("feature_std does not match feature_dim")
        if action_mean.numel() != int(action_dim):
            raise ValueError("action_mean does not match action_dim")
        if action_std.numel() != int(action_dim):
            raise ValueError("action_std does not match action_dim")
        if torch.any(feature_std <= 0) or torch.any(action_std <= 0):
            raise ValueError("critic normalization standard deviations must be positive")
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.register_buffer("action_mean", action_mean)
        self.register_buffer("action_std", action_std)

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def encode_rgb_history_raw(
        self,
        observation_history: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode ``[B, prefix, obs_horizon, ...]`` RGB/robot observations."""
        if set(self.observation_shapes).difference(observation_history):
            missing = sorted(set(self.observation_shapes).difference(observation_history))
            raise KeyError(f"critic RGB input is missing observation keys {missing}")
        first = observation_history[next(iter(self.observation_shapes))]
        if first.ndim < 3:
            raise ValueError("critic observations require batch, prefix, and frame axes")
        batch_size, prefix_length, observation_horizon = first.shape[:3]
        if observation_horizon != self.observation_horizon:
            raise ValueError(
                f"expected observation_horizon={self.observation_horizon}, "
                f"got {observation_horizon}"
            )
        flattened = {}
        for key, shape in self.observation_shapes.items():
            value = observation_history[key]
            expected = (batch_size, prefix_length, observation_horizon, *shape)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"critic observation {key} shape={tuple(value.shape)}; "
                    f"expected={expected}"
                )
            flattened[key] = value.reshape(
                batch_size * prefix_length * observation_horizon,
                *shape,
            )
        frame_features = self.rgb_encoder(obs=flattened)
        frame_features = frame_features.reshape(
            batch_size,
            prefix_length,
            observation_horizon,
            -1,
        )
        return frame_features.flatten(start_dim=2)

    def encode_rgb_history(
        self,
        observation_history: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode and normalize ``[B, prefix, obs_horizon, ...]`` inputs."""
        return self.normalize_features(
            self.encode_rgb_history_raw(observation_history)
        )

    def encode_rgb_boundary(
        self,
        observation_window: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Encode one boundary window shaped ``[B, obs_horizon, ...]``."""
        history = {
            key: value.unsqueeze(1) for key, value in observation_window.items()
        }
        return self.encode_rgb_history(history).squeeze(1)

    def forward_rgb(
        self,
        observation_history: dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ):
        features = self.encode_rgb_history(observation_history)
        actions = self.normalize_actions(actions)
        return super().forward(features, actions, action_mask)


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
    rgb_encoder: nn.Module | None = None,
    observation_shapes: dict[str, tuple[int, ...]] | None = None,
    observation_horizon: int | None = None,
    feature_mean=None,
    feature_std=None,
    action_mean=None,
    action_std=None,
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
    if model_arch in ("v3", "successor_v3", "causal_successor_critic_v3"):
        return CausalSuccessorCriticV3(
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
    if model_arch in ("v4", "rgb_successor_v4", "rgb_successor_critic_v4"):
        required = {
            "rgb_encoder": rgb_encoder,
            "observation_shapes": observation_shapes,
            "observation_horizon": observation_horizon,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "action_mean": action_mean,
            "action_std": action_std,
        }
        missing = sorted(key for key, value in required.items() if value is None)
        if missing:
            raise ValueError(f"model_arch=v4 requires {missing}")
        return RGBSuccessorCriticV4(
            rgb_encoder=rgb_encoder,
            observation_shapes=observation_shapes,
            observation_horizon=observation_horizon,
            feature_dim=feature_dim,
            prediction_horizon=prediction_horizon,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_hidden_dim=action_hidden_dim,
            dropout=dropout,
            feature_mean=feature_mean,
            feature_std=feature_std,
            action_mean=action_mean,
            action_std=action_std,
            action_num_heads=action_num_heads,
            action_conv_layers=action_conv_layers,
            prefix_conv_layers=prefix_conv_layers,
        )
