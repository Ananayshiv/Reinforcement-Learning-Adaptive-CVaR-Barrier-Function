# Observation → PPO → DiffCVaR-CBF-QP Architecture Trace

## Observation representation

`SocialNav` exposes an absolute observation with a fixed maximum number of human
slots. Unused human slots are zero-padded and carry `mask=0`.

The PPO rollout converts the absolute observation to the relative policy format
and then keeps `obs_top_k` human blocks:

```text
absolute environment observation
        |
        v
absolute_obs_batch_to_relative
        |
        v
[robot/goal(6), K * (rel_x, rel_y, vx, vy, radius, mask)]
        |
        v
select_top_k_obs
        |
        v
fixed policy observation
```

The six robot/goal fields are:

```text
goal_rel_x, goal_rel_y, robot_vx, robot_vy, robot_theta, robot_radius
```

Each human block contains:

```text
relative position (2), velocity (2), radius (1), validity mask (1)
```

## Existing DiffCVaR learned path

`DiffCVaRBFQP.forward()` uses one shared learned trunk followed by three heads:

```text
obs
 |
 v
fc1
 |
 +----------------+----------------+
 |                |                |
 v                v                v
fc21             fc22             fc23
 |                |                |
 v                v                v
fc31             fc32             fc33
 |                |                |
u_nom            beta        learned safe radius
```

The returned policy action is not simply `u_nom`. The model passes `u_nom`, the
learned risk level, the learned safety radius, and the physical observation into
the appropriate CVaR-CBF-QP.

## Explicit safety-state path

`_extract_obstacle_blocks()` parses the relative observation into:

```text
rel  : (B, K, 2)
vel  : (B, K, 2)
mask : (B, K)
```

For the single-integrator case, the QP objective is equivalent to keeping the
safe control close to the nominal learned control while satisfying the robust
CVaR-CBF constraints for the valid obstacle slots.

## Graph integration boundary

The learned representation and the safety-constraint state have different
requirements. A learned latent can aggregate a variable set of humans, but the
QP should retain explicit relative geometry, velocity and mask information.

The first graph integration therefore uses two paths:

```text
raw relative observation
        |
        +-------------------------------+
        |                               |
        v                               v
NavigationGraphEncoder          obstacle extraction
        |                               |
        v                               |
fixed graph latent                    CBF state
        |                               |
        v                               |
learned DiffCVaR heads                 |
  |       |       |                     |
u_nom   beta   r_safe                   |
  |       |       |                     |
  +-------+-------+---------------------+
                    |
                    v
                 CVaR-QP
                    |
                    v
                  u_safe
```

The existing critic remains unchanged in this first integration so the actor
representation change can be isolated before modifying both actor and critic.
