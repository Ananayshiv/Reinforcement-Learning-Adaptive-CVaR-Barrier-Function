import numpy as np
import torch

from model.diff_cvar import DiffCVaRBFQP
from model.graph_encoder import NavigationGraphEncoder


class GraphDiffCVaRBFQP(DiffCVaRBFQP):
    """DiffCVaR-CBF-QP policy with a graph-attention learned representation.

    The model deliberately keeps two data paths:

    1. ``raw_obs -> graph encoder -> learned heads`` for ``u_nom``, ``beta``
       and the learned safety radius.
    2. ``raw_obs -> obstacle extraction -> CBF/QP`` for explicit geometry,
       velocities and validity masks.

    Keeping the physical observation available to the QP avoids asking a
    learned latent representation to reconstruct quantities required by the
    safety constraints.
    """

    def __init__(
        self,
        n_features,
        action_dim,
        graph_latent_dim=128,
        graph_human_hidden_dim=64,
        graph_robot_goal_hidden_dim=64,
        graph_dropout=0.0,
        **kwargs,
    ):
        self.raw_n_features = int(n_features)

        # Reuse the existing DiffCVaR learned heads and QP implementation, but
        # size the shared learned trunk for the graph latent instead of the raw
        # flattened observation.
        super().__init__(
            n_features=int(graph_latent_dim),
            action_dim=action_dim,
            **kwargs,
        )

        self.graph_encoder = NavigationGraphEncoder(
            human_hidden_dim=graph_human_hidden_dim,
            robot_goal_hidden_dim=graph_robot_goal_hidden_dim,
            latent_dim=graph_latent_dim,
            dropout=graph_dropout,
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.to(self.fc1.weight.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        raw_obs = obs.reshape(obs.size(0), -1)
        if raw_obs.size(1) != self.raw_n_features:
            raise ValueError(
                f"Expected raw observation width {self.raw_n_features}, "
                f"got {raw_obs.size(1)}"
            )

        # Learned representation path.
        z = self.graph_encoder(raw_obs)
        x = self.act(self.fc1(z))
        x21 = self.act(self.fc21(x))
        x22 = self.act(self.fc22(x))
        x23 = self.act(self.fc23(x))

        u_nom = self.fc31(x21)

        beta_raw = torch.sigmoid(self.fc32(x22)).squeeze(-1)
        beta = self.beta_min + (self.beta - self.beta_min) * beta_raw
        self.last_beta = beta

        r_scale = 1.0 + 1.5 * torch.sigmoid(self.fc33(x23)).squeeze(-1)
        r_safe_learned = self.safe_dist * r_scale
        self.last_r_safe = r_safe_learned

        # Safety path: keep the original physical observation unchanged.
        if self.robot_type == "single_integrator":
            return self._solve_single_integrator_qp(
                raw_obs, u_nom, beta, r_safe_learned
            )
        if self.robot_type == "unicycle":
            return self._solve_unicycle_qp(
                raw_obs, u_nom, beta, r_safe_learned
            )
        if self.robot_type == "unicycle_dynamic":
            raise NotImplementedError("UnicycleDynamic QP is not implemented")
        raise NotImplementedError(
            f"Robot type {self.robot_type} not supported in GraphDiffCVaRBFQP"
        )
