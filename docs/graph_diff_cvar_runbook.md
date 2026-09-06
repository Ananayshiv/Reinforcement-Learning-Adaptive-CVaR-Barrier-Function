# Graph DiffCVaR Review Runbook

## Branch

```text
feature/graph-observation-encoder
```

## Standalone checks

From the repository root:

```bash
python scripts/test_graph_encoder.py
python scripts/test_graph_policy.py
```

The first script validates representation invariants. The second verifies that
GraphDiffCVaR uses the graph latent for the learned policy path while forwarding
the original physical observation to the safety solver.

## Training entrypoint

The new policy is registered as the Hydra model option:

```text
graph_diff_cvar
```

A smoke run can be launched through the existing PPO runner:

```bash
MODEL=graph_diff_cvar WANDB_MODE=disabled NUM_ENVS=1 \
  bash scripts/run_ppo.sh trainer.total_timesteps=1000
```

The current PPO preprocessing still uses `env.obs_top_k`. For an initial
variable-crowd representation experiment, set that value to the number of
available padded human slots so the graph encoder can attend across the full
observation prefix, for example:

```bash
MODEL=graph_diff_cvar WANDB_MODE=disabled NUM_ENVS=1 \
  bash scripts/run_ppo.sh env.obs_top_k=20 trainer.total_timesteps=1000
```

This first integration intentionally leaves the critic on the existing fixed
relative observation. The controlled comparison is therefore focused on the
actor / DiffCVaR learned representation before changing both networks.

## Initial comparison plan

Keep environment, seed, PPO hyperparameters and safety layer fixed. Compare:

1. `diff_cvar` with the existing flattened top-K representation.
2. `graph_diff_cvar` with masked graph aggregation.

Primary metrics already exposed by the training/evaluation code:

- mean return,
- success rate,
- collision rate,
- timeout rate,
- QP infeasibility rate,
- barrier statistics / training stability diagnostics.

A useful follow-up experiment is to train at one crowd density and evaluate at
larger human counts while keeping all non-representation settings fixed.
