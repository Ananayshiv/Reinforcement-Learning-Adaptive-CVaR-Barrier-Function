"""Small end-to-end GraphDiffCVaR PPO smoke test.

This runner deliberately bypasses Hydra's config file loader so the training
path can be validated on Windows while still using the repository's YAML
configuration files through OmegaConf.
"""

from pathlib import Path
import os
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.ppo_trainer import PPOTrainer


OmegaConf.register_new_resolver("math", lambda expr: eval(str(expr)), replace=True)


def load_cfg():
    config_dir = REPO_ROOT / "config"

    base = OmegaConf.load(config_dir / "base.yaml")
    env_cfg = OmegaConf.create(
        {"env": OmegaConf.to_container(OmegaConf.load(config_dir / "env" / "social_nav_var_num.yaml"), resolve=False)}
    )
    # single_integrator.yaml is intentionally packaged at the global level and
    # therefore contains both `robot` and an `env.reward` override.
    robot_cfg = OmegaConf.load(config_dir / "robot" / "single_integrator.yaml")
    model_cfg = OmegaConf.create(
        {"model": OmegaConf.to_container(OmegaConf.load(config_dir / "model" / "graph_diff_cvar.yaml"), resolve=False)}
    )
    trainer_cfg = OmegaConf.create(
        {"trainer": OmegaConf.to_container(OmegaConf.load(config_dir / "trainer" / "ppo.yaml"), resolve=False)}
    )

    cfg = OmegaConf.merge(base, env_cfg, robot_cfg, model_cfg, trainer_cfg)

    # Tiny deterministic smoke settings: enough to exercise rollout, qpth QP,
    # PPO loss/backprop and optimizer update without running a real experiment.
    cfg.device = "cpu"
    cfg.trainer.num_envs = 1
    cfg.trainer.total_timesteps = 128
    cfg.trainer.timesteps_per_batch = 128
    cfg.trainer.num_minibatches = 4
    cfg.trainer.n_updates_per_iteration = 1

    # Keep this smoke focused on the training/qpth path. CVXOPT evaluation can
    # be validated separately once a Windows-compatible CVXOPT install exists.
    cfg.trainer.eval_interval = 0
    cfg.trainer.eval_freq_timesteps = 0
    cfg.trainer.eval_episodes = 1

    return cfg


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg()
    print("Starting GraphDiffCVaR PPO smoke test", flush=True)
    print(f"Resolved model obs_dim: {OmegaConf.to_container(cfg.model, resolve=True)['obs_dim']}", flush=True)
    trainer = PPOTrainer(cfg)
    trainer.train()
    print("GraphDiffCVaR PPO smoke test passed.", flush=True)


if __name__ == "__main__":
    main()
