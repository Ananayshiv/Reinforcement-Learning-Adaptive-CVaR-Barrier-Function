# Graph Observation Encoder Design

## Motivation

The existing policy receives a fixed number of nearest human slots after
`select_top_k_obs`. This is convenient for an MLP but ties the learned
representation to slot count and discards humans outside the selected prefix.

The first graph-based step is to aggregate the valid human set into a
fixed-dimensional latent before the DiffCVaR learned heads.

## Encoder

Input layout:

```text
robot/goal block:
[goal_rel_x, goal_rel_y, rvx, rvy, rtheta, robot_radius]

human block:
[rel_x, rel_y, human_vx, human_vy, human_radius, mask]
```

Architecture:

```text
robot/goal(6) -> robot_goal_encoder ---------------------+
                                                        |
human_i(5) -> shared human_encoder -> key/value --------+-> fusion -> latent
                             ^                          |
                             |                          |
                 robot/goal query -> masked attention -+
```

The same human encoder is reused for every slot. Attention is conditioned on
the robot/goal context and normalized only across valid human slots.

## Mask handling

The repository already carries a validity mask for every human slot. Invalid
slots are removed before attention normalization. The all-invalid case maps to
a zero crowd context instead of producing a softmax NaN.

## Invariants covered by tests

- output width is independent of the number of human slots,
- invalid slots receive zero attention,
- changing the contents of masked slots does not change the latent,
- reordering human slots does not change the aggregated representation,
- zero-human and all-invalid cases remain finite,
- gradients propagate through the encoder.

## Integration choice

The graph latent is used only for the learned DiffCVaR heads in the first
version. The raw relative observation is retained for CBF/QP constraint
construction. This keeps the safety representation explicit and makes the
first experiment easier to interpret.
