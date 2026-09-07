from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from model.graph_diff_cvar import GraphDiffCVaRBFQP


def make_relative_obs(batch=3, humans=5):
    obs = torch.zeros(batch, 6 + 6 * humans, dtype=torch.float32)
    obs[:, 0] = -3.0
    obs[:, 1] = 1.0
    obs[:, 2] = 0.2
    obs[:, 3] = -0.1
    obs[:, 4] = 0.3
    obs[:, 5] = 0.3

    blocks = obs[:, 6:].reshape(batch, humans, 6)
    for i in range(humans):
        blocks[:, i, 0] = 0.75 + 0.2 * i
        blocks[:, i, 1] = -0.15 * i
        blocks[:, i, 2] = -0.1
        blocks[:, i, 3] = 0.1
        blocks[:, i, 4] = 0.3
        blocks[:, i, 5] = 1.0
    return obs


def test_graph_policy_keeps_raw_obs_for_safety_path():
    obs = make_relative_obs(batch=3, humans=5)
    model = GraphDiffCVaRBFQP(
        n_features=obs.shape[1],
        action_dim=2,
        graph_latent_dim=32,
        graph_human_hidden_dim=16,
        graph_robot_goal_hidden_dim=16,
        hidden_dim=32,
        control_hidden_dim=16,
        scalar_hidden_dim=16,
        robot_type="single_integrator",
        safe_dist=0.8,
        alpha=2.0,
        beta_min=0.05,
        beta=0.5,
        vmax=1.0,
        omega_max=1.0,
        gmm_weights=[0.6, 0.2, 0.2],
        gmm_stds=[0.1, 0.2, 0.2],
        gmm_lateral_ratio=0.3,
    )

    captured = {}

    def fake_solver(raw_obs, u_nom, beta, r_safe):
        captured["raw_obs"] = raw_obs.detach().clone()
        captured["u_nom"] = u_nom.detach().clone()
        captured["beta"] = beta.detach().clone()
        captured["r_safe"] = r_safe.detach().clone()
        return u_nom

    model._solve_single_integrator_qp = fake_solver
    out = model(obs)

    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()
    assert torch.equal(captured["raw_obs"], obs)
    assert captured["u_nom"].shape == (3, 2)
    assert captured["beta"].shape == (3,)
    assert captured["r_safe"].shape == (3,)


def test_graph_encoder_receives_gradients_through_policy_heads():
    obs = make_relative_obs(batch=2, humans=4)
    model = GraphDiffCVaRBFQP(
        n_features=obs.shape[1],
        action_dim=2,
        graph_latent_dim=32,
        graph_human_hidden_dim=16,
        graph_robot_goal_hidden_dim=16,
        hidden_dim=32,
        control_hidden_dim=16,
        scalar_hidden_dim=16,
        robot_type="single_integrator",
        safe_dist=0.8,
        alpha=2.0,
        beta_min=0.05,
        beta=0.5,
        vmax=1.0,
        omega_max=1.0,
        gmm_weights=[0.6, 0.2, 0.2],
        gmm_stds=[0.1, 0.2, 0.2],
        gmm_lateral_ratio=0.3,
    )

    # Bypass the numerical QP for this unit test while retaining the learned
    # nominal-control path in the autograd graph.
    model._solve_single_integrator_qp = lambda raw_obs, u_nom, beta, r_safe: u_nom

    loss = model(obs).pow(2).mean()
    loss.backward()

    graph_grads = [
        p.grad for p in model.graph_encoder.parameters() if p.requires_grad
    ]
    assert graph_grads
    assert any(g is not None and torch.isfinite(g).all() for g in graph_grads)


if __name__ == "__main__":
    test_graph_policy_keeps_raw_obs_for_safety_path()
    test_graph_encoder_receives_gradients_through_policy_heads()
    print("All GraphDiffCVaR policy integration tests passed.")
