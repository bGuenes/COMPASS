import torch
import numpy as np

# --------------------------------------------------------------------------------------------------
# Stochastic Differential Equations

#################################################################################################
# ///////////////////////////// Stochastic Differential Equations ///////////////////////////////
#################################################################################################

"""
    Stochastic Differential Equations (SDEs) for diffusion models.
    - VESDE: Variance Exploding SDE
    - VPSDE: Variance Preserving SDE

    Common interface (perturbation kernel p_0t(x_t | x_0) = N(alpha_t * x_0, sigma_t^2)):
        alpha_t(t)          -- mean scaling alpha(t)   (VESDE: 1)
        marginal_prob_std(t)-- noise std sigma(t)
        sigma_t(t)          -- alias of marginal_prob_std
        time_of_sigma(s)    -- inverse of sigma(t)
        lambda_t(t)         -- noise-to-signal scale lambda(t) = sigma(t) / alpha(t)
        time_of_lambda(l)   -- inverse of lambda(t)

    All samplers and the probability-flow ODE are written in terms of the
    rescaled state y = x / alpha(t) and the noise scale lambda(t): in these
    variables every SDE here becomes variance-exploding with dy = sqrt(d lambda^2/dt) dW,
    so a single (y, lambda)-space integrator covers both SDE types.
    For the VESDE (alpha = 1, lambda = sigma) this reduces to the previous behavior.
"""

#############################################
# ----- VESDE -----
#############################################
class VESDE():
    def __init__(self, sigma=25.0):
        """
        Variance Exploding Stochastic Differential Equation (VESDE) class.
        The VESDE is defined as:
            Drift     -> f(x,t) = 0
            Diffusion -> g(t)   = sigma^t
        """
        self.sigma = torch.tensor(sigma)

    def alpha_t(self, t):
        """Mean scaling of the perturbation kernel; identically 1 for the VESDE."""
        t = torch.as_tensor(t)
        return torch.ones_like(t)

    def marginal_prob_std(self, t):
        """
        Compute the standard deviation of p_{0t}(x(t) | x(0)) for VESDE.

        Args:
            t: A tensor of time steps.
        Returns:
            The standard deviation.
        """
        try:
            return torch.sqrt((self.sigma ** (2 * t) - 1.0) / (2 * torch.log(self.sigma)))
        except:
            return torch.sqrt((self.sigma ** (2 * t) - 1.0) / (2 * np.log(self.sigma)))

    def sigma_t(self, t):
        """
        Compute sigma_t (noise standard deviation).
        """
        return self.marginal_prob_std(t)

    def time_of_sigma(self, sigma):
        """
        Inverse of marginal_prob_std: the diffusion time t at which the marginal
        noise standard deviation equals `sigma`.
        """
        return torch.log(1.0 + 2.0 * torch.log(self.sigma) * sigma**2) / (2.0 * torch.log(self.sigma))

    def lambda_t(self, t):
        """Noise-to-signal scale lambda(t) = sigma(t)/alpha(t); equals sigma(t) for the VESDE."""
        return self.marginal_prob_std(t)

    def time_of_lambda(self, lam):
        """Inverse of lambda_t; equals time_of_sigma for the VESDE."""
        return self.time_of_sigma(lam)

#############################################
# ----- VPSDE -----
#############################################
class VPSDE():
    def __init__(self, beta_min=0.1, beta_max=20.0):
        """
        Variance Preserving Stochastic Differential Equation (VPSDE, Song et al. 2021)
        with the linear schedule beta(t) = beta_min + t * (beta_max - beta_min):
            Drift     -> f(x,t) = -1/2 beta(t) x
            Diffusion -> g(t)   = sqrt(beta(t))

        Perturbation kernel: p_0t(x_t|x_0) = N(alpha(t) x_0, sigma(t)^2) with
            B(t)     = beta_min t + 1/2 (beta_max - beta_min) t^2   (= int_0^t beta)
            alpha(t) = exp(-B(t)/2)
            sigma(t) = sqrt(1 - exp(-B(t)))
        """
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)

    def _B(self, t):
        t = torch.as_tensor(t)
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def beta_t(self, t):
        t = torch.as_tensor(t)
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t):
        """Mean scaling alpha(t) = exp(-B(t)/2)."""
        return torch.exp(-0.5 * self._B(t))

    def marginal_prob_std(self, t):
        """Noise std sigma(t) = sqrt(1 - exp(-B(t))) of p_{0t}(x(t)|x(0))."""
        return torch.sqrt(1.0 - torch.exp(-self._B(t)))

    def sigma_t(self, t):
        return self.marginal_prob_std(t)

    def _time_of_B(self, B):
        """Solve B(t) = B for t (quadratic in t, positive root)."""
        db = self.beta_max - self.beta_min
        return (torch.sqrt(self.beta_min**2 + 2.0 * db * B) - self.beta_min) / db

    def time_of_sigma(self, sigma):
        """Inverse of marginal_prob_std (sigma must be < 1)."""
        sigma = torch.as_tensor(sigma)
        B = -torch.log1p(-sigma**2)
        return self._time_of_B(B)

    def lambda_t(self, t):
        """Noise-to-signal scale lambda(t) = sigma(t)/alpha(t) = sqrt(exp(B(t)) - 1)."""
        return torch.sqrt(torch.expm1(self._B(t)))

    def time_of_lambda(self, lam):
        """Inverse of lambda_t."""
        lam = torch.as_tensor(lam)
        B = torch.log1p(lam**2)
        return self._time_of_B(B)
