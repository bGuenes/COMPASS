"""
Analytic verification of the probability-flow-ODE log-probability, the
score-ascent MAP and the VPSDE implementation.

As in test_multiobs_analytic.py, a mock "score network" returns the EXACT score
of the diffused posterior for a linear-Gaussian toy model, so any error is
attributable to the PF-ODE / sampler machinery, not to training.

Toy model: theta ~ N(MU0, SIG0^2), x | theta ~ N(theta, SIGX^2)  (all diagonal)
    posterior:  theta | x ~ N(m, v)  with  1/v = 1/SIG0^2 + 1/SIGX^2,
                m = v * (MU0/SIG0^2 + x/SIGX^2)

Runs on CPU in a few minutes:
    python tests/test_pfode_vpsde_analytic.py
or with pytest:
    pytest tests/test_pfode_vpsde_analytic.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from compass.PFODE import PFODE
from compass.Sampler import Sampler
from compass.SDE import VESDE, VPSDE

MU0 = torch.tensor([-2.3, -2.89])
SIG0 = torch.tensor([0.3, 0.3])
SIGX = torch.tensor([0.5, 0.5])

LAM = 1 / SIG0**2 + 1 / SIGX**2      # posterior precision
V = 1 / LAM                          # posterior variance


def posterior_moments(x_obs):
    m = (MU0 / SIG0**2 + x_obs / SIGX**2) / LAM
    return m, V


class MockSBIm:
    """Minimal stand-in for ScoreBasedInferenceModel."""
    def __init__(self, model_cls, sde, nodes=4):
        self.sde = sde
        self.model = model_cls(sde)
        self.sampler = Sampler(self)
        self.nodes_size = nodes

    def output_scale_function(self, t, x):
        return x / self.sde.marginal_prob_std(t).to(x.device)


class DiffusedPosteriorScore(torch.nn.Module):
    """Nodes [th1, th2, x1, x2]; returns sigma_t * (exact x-space score of the
    diffused posterior) for a general SDE with perturbation kernel
    N(alpha_t theta_0, sigma_t^2):  p_t(theta_t | x) = N(alpha m, alpha^2 v + sigma^2).
    Matches the network convention model_out = sigma_t * s_x."""
    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    def forward(self, x, t, c, return_attn_weights=False):
        sigma_t = self.sde.marginal_prob_std(t).to(x.device)
        alpha_t = self.sde.alpha_t(t).to(x.device)
        theta_t, x_obs = x[:, :2], x[:, 2:]
        m, v = posterior_moments(x_obs)
        score = -(theta_t - alpha_t * m) / (alpha_t**2 * v + sigma_t**2)
        out = torch.zeros_like(x)
        out[:, :2] = sigma_t * score
        if return_attn_weights:
            return out, torch.zeros(1)
        return out


def analytic_logpdf(theta, m, var):
    """log N(theta; m, var) with diagonal var, summed over dims."""
    return (-0.5 * torch.log(2 * torch.pi * var)
            - (theta - m)**2 / (2 * var)).sum(-1)


def _eval_points(m, s, n=7):
    """Evaluation grid: mode and +-1, +-2 sigma offsets along both dims."""
    offs = torch.tensor([0., 1., -1., 2., -2., 0.5, -1.5])[:n]
    return m + offs.unsqueeze(1) * s


MASK = torch.tensor([0., 0., 1., 1.])


def _pfode_logprob(sde, timesteps=100, eps=1e-3):
    torch.manual_seed(0)
    x_obs = MU0 + 0.4 * torch.randn(2)
    m, v = posterior_moments(x_obs)
    s = torch.sqrt(v)

    sbim = MockSBIm(DiffusedPosteriorScore, sde)
    pfode = PFODE(sbim)

    pts = _eval_points(m, s)
    data = torch.cat([pts, x_obs.repeat(len(pts), 1)], dim=1)
    lp = pfode.log_prob(data, MASK, timesteps=timesteps, eps=eps)

    sig_eps = sde.sigma_t(torch.tensor(eps))
    alpha_eps = sde.alpha_t(torch.tensor(eps))
    # PF-ODE returns the eps-smoothed density N(alpha m, alpha^2 v + sigma_eps^2)
    lp_true = analytic_logpdf(pts, alpha_eps * m, alpha_eps**2 * v + sig_eps**2)
    return lp, lp_true


def test_pfode_logprob_vesde():
    lp, lp_true = _pfode_logprob(VESDE(sigma=25.0))
    abs_err = (lp - lp_true).abs()
    # Differences between evaluation points must be almost exact (the constant
    # Gaussian-prior approximation at t=1 cancels)
    rel = (lp - lp[0]) - (lp_true - lp_true[0])
    print("VESDE  abs err:", abs_err.tolist())
    print("VESDE  rel err:", rel.abs().tolist())
    assert abs_err.max() < 0.15, f"absolute log-prob error {abs_err.max():.3f}"
    assert rel.abs().max() < 0.02, f"relative log-prob error {rel.abs().max():.4f}"


def test_pfode_logprob_vesde_hutchinson():
    torch.manual_seed(1)
    sde = VESDE(sigma=25.0)
    x_obs = MU0 + 0.4 * torch.randn(2)
    m, v = posterior_moments(x_obs)
    sbim = MockSBIm(DiffusedPosteriorScore, sde)
    pfode = PFODE(sbim)
    pts = _eval_points(m, torch.sqrt(v), n=3)
    data = torch.cat([pts, x_obs.repeat(len(pts), 1)], dim=1)
    lp_exact = pfode.log_prob(data, MASK, timesteps=100)
    lp_hutch = pfode.log_prob(data, MASK, timesteps=100,
                              divergence="hutchinson", hutchinson_samples=64)
    err = (lp_exact - lp_hutch).abs()
    print("hutchinson vs exact:", err.tolist())
    assert err.max() < 0.15


def test_pfode_logprob_vpsde():
    lp, lp_true = _pfode_logprob(VPSDE())
    abs_err = (lp - lp_true).abs()
    rel = (lp - lp[0]) - (lp_true - lp_true[0])
    print("VPSDE  abs err:", abs_err.tolist())
    print("VPSDE  rel err:", rel.abs().tolist())
    assert abs_err.max() < 0.15, f"absolute log-prob error {abs_err.max():.3f}"
    assert rel.abs().max() < 0.02, f"relative log-prob error {rel.abs().max():.4f}"


def test_map_estimate():
    torch.manual_seed(2)
    for sde in [VESDE(sigma=25.0), VPSDE()]:
        x_obs = MU0 + 0.4 * torch.randn(2)
        m, v = posterior_moments(x_obs)
        sbim = MockSBIm(DiffusedPosteriorScore, sde)
        pfode = PFODE(sbim)
        # Start the ascent well away from the mode
        init = m + 2.0 * torch.sqrt(v)
        data = torch.cat([init, x_obs]).unsqueeze(0)
        z = pfode.map_estimate(data, MASK, sigma_start=1.0, timesteps=100)
        err = (z[0, :2] - m).abs() / torch.sqrt(v)
        print(f"{type(sde).__name__} MAP err (in sigma): {err.tolist()}")
        assert err.max() < 0.05, f"MAP off by {err.max():.3f} posterior sigma"


def _sample_posterior(sde, method, **kw):
    torch.manual_seed(3)
    x_obs = MU0 + 0.4 * torch.randn(2)
    m, v = posterior_moments(x_obs)
    sbim = MockSBIm(DiffusedPosteriorScore, sde)
    samples = sbim.sampler.sample(
        world_size=1, data=x_obs, condition_mask=MASK,
        timesteps=50, num_samples=4000,
        method=method, device="cpu", verbose=False, **kw)
    th = samples[0, :, :2]
    mean_err = (th.mean(0) - m).abs() / torch.sqrt(v)
    std_ratio = th.std(0) / torch.sqrt(v)
    return mean_err, std_ratio


def test_vpsde_sampler_dpm():
    mean_err, std_ratio = _sample_posterior(VPSDE(), "dpm", order=2)
    print(f"VPSDE dpm: mean err {mean_err.tolist()} sigma, std ratio {std_ratio.tolist()}")
    assert mean_err.max() < 0.10
    assert ((std_ratio > 0.9) & (std_ratio < 1.1)).all()


def test_vpsde_sampler_euler():
    mean_err, std_ratio = _sample_posterior(VPSDE(), "euler")
    print(f"VPSDE euler: mean err {mean_err.tolist()} sigma, std ratio {std_ratio.tolist()}")
    assert mean_err.max() < 0.10
    assert ((std_ratio > 0.85) & (std_ratio < 1.15)).all()


def test_vesde_sampler_regression():
    # The (y, lambda)-space generalization must leave VESDE sampling unchanged
    mean_err, std_ratio = _sample_posterior(VESDE(sigma=25.0), "dpm", order=2)
    print(f"VESDE dpm: mean err {mean_err.tolist()} sigma, std ratio {std_ratio.tolist()}")
    assert mean_err.max() < 0.10
    assert ((std_ratio > 0.9) & (std_ratio < 1.1)).all()


def test_vpsde_end_to_end_training():
    """Train a tiny VPSDE model on the linear-Gaussian joint and check the
    sampled posterior and the PF-ODE likelihood are in the right place."""
    from compass.ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm

    torch.manual_seed(4)
    n_train = 20_000
    theta = MU0 + SIG0 * torch.randn(n_train, 2)
    x = theta + SIGX * torch.randn(n_train, 2)

    model = SBIm(nodes_size=4, sde_type="vpsde", hidden_size=64, depth=3,
                 num_heads=2, mlp_ratio=2)
    model.train(theta=theta, x=x, batch_size=512, max_epochs=40, lr=1e-3,
                device="cpu", verbose=False, early_stopping_patience=40,
                path="/tmp/compass_vpsde_smoke")

    x_obs = MU0.unsqueeze(0)
    m, v = posterior_moments(x_obs[0])
    samples = model.sample(x=x_obs, timesteps=50, num_samples=2000,
                           device="cpu", verbose=False, method="dpm")
    th = samples[0]
    mean_err = (th.mean(0) - m).abs() / torch.sqrt(v)
    std_ratio = th.std(0) / torch.sqrt(v)
    print(f"VPSDE e2e: mean err {mean_err.tolist()} sigma, std ratio {std_ratio.tolist()}")
    assert mean_err.max() < 0.5, "trained VPSDE posterior mean off"
    assert ((std_ratio > 0.6) & (std_ratio < 1.6)).all(), "trained VPSDE posterior width off"

    # PF-ODE log posterior at the analytic mode vs 2-sigma away: finite and ordered
    pts = torch.stack([m, m + 2 * torch.sqrt(v)])
    data = torch.cat([pts, x_obs.repeat(2, 1)], dim=1)
    lp = model.log_prob(data, MASK, timesteps=64)
    print(f"VPSDE e2e log_prob: mode {lp[0]:.2f}, 2sigma {lp[1]:.2f}")
    assert torch.isfinite(lp).all()
    assert lp[0] > lp[1], "log-prob at mode should exceed log-prob 2 sigma away"


if __name__ == "__main__":
    test_pfode_logprob_vesde()
    test_pfode_logprob_vesde_hutchinson()
    test_pfode_logprob_vpsde()
    test_map_estimate()
    test_vpsde_sampler_dpm()
    test_vpsde_sampler_euler()
    test_vesde_sampler_regression()
    test_vpsde_end_to_end_training()
    print("All PF-ODE / VPSDE analytic tests passed.")
