import torch

from model.graph_encoder import NavigationGraphEncoder


def make_relative_obs(batch=4, humans=5):
    obs = torch.zeros(batch, 6 + 6 * humans, dtype=torch.float32)
    obs[:, 0] = -3.0
    obs[:, 1] = 1.0
    obs[:, 2] = 0.2
    obs[:, 3] = -0.1
    obs[:, 4] = 0.3
    obs[:, 5] = 0.3

    if humans > 0:
        blocks = obs[:, 6:].reshape(batch, humans, 6)
        for i in range(humans):
            blocks[:, i, 0] = 0.5 + i
            blocks[:, i, 1] = -0.2 * i
            blocks[:, i, 2] = -0.1
            blocks[:, i, 3] = 0.15
            blocks[:, i, 4] = 0.3
            blocks[:, i, 5] = 1.0
    return obs


def test_fixed_latent_across_crowd_sizes():
    encoder = NavigationGraphEncoder(latent_dim=32)
    for humans in [0, 1, 2, 5, 15]:
        out = encoder(make_relative_obs(batch=4, humans=humans))
        assert out.shape == (4, 32)


def test_masked_humans_receive_zero_attention():
    encoder = NavigationGraphEncoder(latent_dim=32)
    obs = make_relative_obs(batch=2, humans=5)
    blocks = obs[:, 6:].reshape(2, 5, 6)
    blocks[:, 3:, 5] = 0.0

    _, attention = encoder(obs, return_attention=True)
    assert torch.allclose(attention[:, 3:], torch.zeros_like(attention[:, 3:]))
    assert torch.allclose(attention[:, :3].sum(dim=1), torch.ones(2), atol=1e-6)


def test_all_invalid_humans_are_stable():
    encoder = NavigationGraphEncoder(latent_dim=32)
    obs = make_relative_obs(batch=3, humans=5)
    obs[:, 6:].reshape(3, 5, 6)[:, :, 5] = 0.0

    out, attention = encoder(obs, return_attention=True)
    assert torch.isfinite(out).all()
    assert torch.equal(attention, torch.zeros_like(attention))


def test_zero_human_case():
    encoder = NavigationGraphEncoder(latent_dim=16)
    out, attention = encoder(make_relative_obs(batch=2, humans=0), return_attention=True)
    assert out.shape == (2, 16)
    assert attention.shape == (2, 0)
    assert torch.isfinite(out).all()


def test_masked_slots_do_not_affect_latent():
    torch.manual_seed(0)
    encoder = NavigationGraphEncoder(latent_dim=32)
    encoder.eval()

    obs_a = make_relative_obs(batch=1, humans=5)
    obs_b = obs_a.clone()
    blocks_a = obs_a[:, 6:].reshape(1, 5, 6)
    blocks_b = obs_b[:, 6:].reshape(1, 5, 6)

    blocks_a[:, 3:, 5] = 0.0
    blocks_b[:, 3:, 5] = 0.0
    blocks_b[:, 3:, :5] = 999.0

    assert torch.allclose(encoder(obs_a), encoder(obs_b), atol=1e-6)


def test_permutation_invariance_over_human_slots():
    torch.manual_seed(0)
    encoder = NavigationGraphEncoder(latent_dim=32)
    encoder.eval()

    obs_a = make_relative_obs(batch=1, humans=5)
    obs_b = obs_a.clone()
    blocks = obs_b[:, 6:].reshape(1, 5, 6)
    perm = torch.tensor([4, 2, 0, 3, 1])
    blocks[:] = blocks[:, perm, :].clone()

    assert torch.allclose(encoder(obs_a), encoder(obs_b), atol=1e-6)


def test_gradients_flow():
    encoder = NavigationGraphEncoder(latent_dim=32)
    obs = make_relative_obs(batch=4, humans=5)
    loss = encoder(obs).pow(2).mean()
    loss.backward()

    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


if __name__ == "__main__":
    test_fixed_latent_across_crowd_sizes()
    test_masked_humans_receive_zero_attention()
    test_all_invalid_humans_are_stable()
    test_zero_human_case()
    test_masked_slots_do_not_affect_latent()
    test_permutation_invariance_over_human_slots()
    test_gradients_flow()
    print("All NavigationGraphEncoder tests passed.")
