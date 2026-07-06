"""
Regression test for model comparison with models that have DIFFERENT numbers
of physical parameters (different theta dimensions).

The AIC/AICc used by ``ModelTransfuser.compare`` must penalise the number of
PHYSICAL model parameters (the theta dimension), not the neural-network weight
count. This test uses analytic-score-style mock models (like
tests/test_multiobs_analytic.py) so that the sampling machinery is replaced by
exact Gaussian draws and any error is attributable to the comparison /
information-criterion logic.

Two toy generative models over a 2-D observation x:
    Model A (2 params): x ~ N(theta[:2], sigma^2 I).   theta = (a, b)
    Model B (3 params): x ~ N(theta[:2], sigma^2 I).   theta = (a, b, c)
                         The third parameter c is spurious -- it does not enter
                         the likelihood, so B fits the data just as well as A
                         but pays an extra AICc penalty.
The observations are generated from the simpler model A, which must therefore
win the AICc ranking.

Runs on CPU in well under a minute:
    python tests/test_variable_param_count.py
or with pytest:
    pytest tests/test_variable_param_count.py
"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from compass.ModelTransfuser import ModelTransfuser

# Ground-truth generative parameters
THETA_TRUE = torch.tensor([1.5, -0.7])   # (a, b) mean of x
SIGMA_X = 0.5                            # observation noise std
X_DIM = 2

# Posterior / prior widths for the mocks
TAU = 0.15          # posterior std on the data-constrained parameters
C_PRIOR_MEAN = 0.0  # prior for the spurious parameter of model B
C_PRIOR_STD = 1.0


class MockSBIm:
    """
    Minimal stand-in for ScoreBasedInferenceModel that returns analytic
    Gaussian samples instead of running diffusion sampling.

    theta_dim: number of physical parameters of this model.
    Only the first two theta components ever enter the observation model; any
    further components are spurious (constrained only by their prior).
    """

    def __init__(self, theta_dim, seed=0):
        self.theta_dim = theta_dim
        self.nodes_size = theta_dim + X_DIM
        self.seed = seed
        # compare() reads model.sampler.all_attn_weights after posterior sampling.
        self.sampler = SimpleNamespace(all_attn_weights=torch.zeros(1))

    def sample(self, theta=None, x=None, err=None, condition_mask=None,
               num_samples=1000, **kwargs):
        """
        Posterior sampling  (x given):     returns (n_obs, num_samples, theta_dim)
        Likelihood sampling (theta given): returns (n_obs, num_samples, X_DIM)
        """
        g = torch.Generator().manual_seed(self.seed)

        if x is not None and theta is None:
            # ----- posterior p(theta | x_i) -----
            x = torch.as_tensor(x, dtype=torch.float32)
            n_obs = x.shape[0]
            out = torch.empty(n_obs, num_samples, self.theta_dim)
            for i in range(n_obs):
                # first two dims: data-constrained, centred on the observation
                out[i, :, 0] = x[i, 0] + TAU * torch.randn(num_samples, generator=g)
                out[i, :, 1] = x[i, 1] + TAU * torch.randn(num_samples, generator=g)
                # any extra dims: spurious -> follow the (broad) prior
                for d in range(2, self.theta_dim):
                    out[i, :, d] = C_PRIOR_MEAN + C_PRIOR_STD * torch.randn(num_samples, generator=g)
            return out

        elif theta is not None and x is None:
            # ----- likelihood p(x | theta_i = MAP_i) -----
            theta = torch.as_tensor(theta, dtype=torch.float32)
            n_obs = theta.shape[0]
            out = torch.empty(n_obs, num_samples, X_DIM)
            for i in range(n_obs):
                mean = theta[i, :2]  # only the first two params drive the mean
                out[i, :, 0] = mean[0] + SIGMA_X * torch.randn(num_samples, generator=g)
                out[i, :, 1] = mean[1] + SIGMA_X * torch.randn(num_samples, generator=g)
            return out

        raise ValueError("MockSBIm.sample expects exactly one of x / theta")


def _make_observations(n_obs, seed=1):
    g = torch.Generator().manual_seed(seed)
    return THETA_TRUE + SIGMA_X * torch.randn(n_obs, X_DIM, generator=g)


def _build_mtf():
    mtf = ModelTransfuser()
    mtf.add_model("A_2params", MockSBIm(theta_dim=2, seed=10))
    mtf.add_model("B_3params", MockSBIm(theta_dim=3, seed=20))
    mtf.trained_models = True  # bypass the training check; models are analytic
    return mtf


def test_variable_param_count_end_to_end():
    """(i) compare() runs with models of different theta dims,
       (ii) param_count == theta dim, (iii) the simpler true model wins."""
    torch.manual_seed(0)
    np.random.seed(0)
    x = _make_observations(n_obs=25)

    mtf = _build_mtf()
    mtf.compare(x=x, num_samples=400, device="cpu", verbose=False)

    # (ii) recorded param_count equals the physical theta dimension
    assert mtf.stats["A_2params"]["param_count"] == 2, \
        f"A param_count {mtf.stats['A_2params']['param_count']} != 2"
    assert mtf.stats["B_3params"]["param_count"] == 3, \
        f"B param_count {mtf.stats['B_3params']['param_count']} != 3"

    # sanity: both models fit the data comparably (spurious param does not help)
    ll_a = mtf.stats["A_2params"]["log_probs"].sum().item()
    ll_b = mtf.stats["B_3params"]["log_probs"].sum().item()
    print(f"log-likelihood  A={ll_a:.2f}  B={ll_b:.2f}")
    assert abs(ll_a - ll_b) < 0.25 * abs(ll_a) + 5.0, \
        "models should fit comparably; test setup broken"

    # (iii) the simpler (true) model wins the AICc ranking
    p_a = mtf.stats["A_2params"]["model_prob"]
    p_b = mtf.stats["B_3params"]["model_prob"]
    print(f"model probabilities  A={100*p_a:.2f}%  B={100*p_b:.2f}%")
    assert mtf.stats["A_2params"]["AIC"] < mtf.stats["B_3params"]["AIC"], \
        "simpler model must have the lower AICc"
    assert p_a > p_b, "simpler (true) model must win the AICc ranking"
    assert p_a > 0.9, f"simpler model should dominate, got {100*p_a:.1f}%"

    # obs_probs shape is (n_obs,) per model and the per-obs probabilities are
    # normalised across the two models.
    assert mtf.stats["A_2params"]["obs_probs"].shape == (x.shape[0],)
    total = mtf.stats["A_2params"]["obs_probs"] + mtf.stats["B_3params"]["obs_probs"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-4)


def test_small_sample_guard_no_crash_and_warns():
    """(iv) with n_obs small enough that n_obs - k - 1 <= 0, no crash and a
    warning is printed instead of a garbage (negative-denominator) correction."""
    torch.manual_seed(0)
    np.random.seed(0)
    # n_obs = 3 -> for B (k=3): 3 - 3 - 1 = -1 <= 0  (guard trips)
    #           -> for A (k=2): 3 - 2 - 1 =  0 <= 0  (guard trips)
    x = _make_observations(n_obs=3, seed=2)

    mtf = _build_mtf()
    buf = io.StringIO()
    with redirect_stdout(buf):
        mtf.compare(x=x, num_samples=400, device="cpu", verbose=False)
    out = buf.getvalue()

    # No crash, finite AICc values (fell back to plain AIC).
    for name in ("A_2params", "B_3params"):
        aic = mtf.stats[name]["AIC"]
        assert torch.isfinite(torch.as_tensor(aic)), f"{name} AICc not finite: {aic}"

    assert mtf._aicc_warned_once is True, "guard flag should be set"
    assert "AICc small-sample correction is undefined" in out, \
        f"expected a printed small-sample warning; got:\n{out}"
    print("small-sample guard: warning printed and comparison completed")


if __name__ == "__main__":
    test_variable_param_count_end_to_end()
    test_small_sample_guard_no_crash_and_warns()
    print("All variable-parameter-count model-comparison tests passed.")
