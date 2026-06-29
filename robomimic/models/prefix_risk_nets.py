"""Networks shared by prefix-risk training and policy regularization."""

import torch
import torch.nn as nn


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
