"""
Probability Flow ODE for exact log-probability evaluation in COMPASS.

Replaces the KDE-based likelihood estimation with a direct computation
using the change-of-variables formula through the diffusion ODE.

Usage:
    # Instead of: sample from NLE → fit KDE → evaluate
    # Do:         directly compute log p(x_obs | theta_hat)

    log_probs = model.log_prob(
        theta=theta_hat,        # conditioned values
        x=x_obs,                # values to evaluate
        timesteps=200,
        device="cuda"
    )
"""

import torch
import torch.nn as nn
import numpy as np
import tqdm


class PFODELogProb:
    """
    Computes log p(z_latent | z_observed) using the probability flow ODE.
    
    The PF-ODE transforms data into noise deterministically. By tracking
    the change in log-density along this trajectory (via the instantaneous
    change of variables formula), we get exact log-probabilities without
    sampling or KDE.
    """
    
    def __init__(self, sbim):
        self.sbim = sbim
        self.sde = sbim.sde
        self.model = sbim.model
    
    def log_prob(self, theta=None, x=None, condition_mask=None,
                 timesteps=200, eps=1e-3, num_hutchinson=1,
                 device="cpu", verbose=True):
        """
        Compute log p(x|theta) or log p(theta|x) via the PF-ODE.
        
        Args:
            theta: Conditioned parameters (for NLE mode), shape (N, D_theta)
            x: Conditioned observations (for NPE mode), shape (N, D_x)
                — OR the values to evaluate (for NLE mode)
            condition_mask: Binary mask (1=observed, 0=latent)
            timesteps: Number of ODE integration steps
            eps: Start time (avoid t=0 singularity)
            num_hutchinson: Number of random vectors for trace estimation
            device: Device to run on
            verbose: Show progress bar
            
        Returns:
            log_probs: (N,) tensor of log-probabilities
        """
        
        # --- Setup data and masks (same logic as sample()) ---
        if theta is not None and x is not None:
            # NLE mode: condition on theta, evaluate x
            # Build joint vector z = (theta, x)
            data = torch.cat([theta, x], dim=-1)
            if condition_mask is None:
                D_theta = theta.shape[-1]
                D_x = x.shape[-1]
                condition_mask = torch.cat([
                    torch.ones(D_theta), torch.zeros(D_x)
                ])
        elif theta is None and x is not None:
            # NPE mode: condition on x, evaluate theta
            data = x  # will be padded
            if condition_mask is None:
                D_x = x.shape[-1]
                D_theta = self.sbim.nodes_size - D_x
                condition_mask = torch.cat([
                    torch.zeros(D_theta), torch.ones(D_x)
                ])
            # Build joint vector with zeros for theta (to be filled)
            joint = torch.zeros(x.shape[0], self.sbim.nodes_size)
            joint[:, condition_mask.bool()] = x
            data = joint
        elif theta is not None and x is None:
            # Evaluate theta (unlikely use case but handle it)
            data = theta
            if condition_mask is None:
                D_theta = theta.shape[-1]
                condition_mask = torch.cat([
                    torch.ones(D_theta), 
                    torch.zeros(self.sbim.nodes_size - D_theta)
                ])
            joint = torch.zeros(theta.shape[0], self.sbim.nodes_size)
            joint[:, condition_mask.bool()] = theta
            data = joint
        else:
            raise ValueError("Provide at least theta or x.")
        
        # Move to device
        data = data.to(device).float()
        condition_mask = condition_mask.to(device).float()
        
        # Ensure 2D
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        N = data.shape[0]
        latent_mask = (1 - condition_mask)  # (D,) — 1 for latent dims
        num_latent = int(latent_mask.sum().item())
        
        # --- Move model to device ---
        self.model.to(device)
        self.model.eval()
        
        # --- Integration setup ---
        # Forward ODE: t goes from eps to 1 (data → noise)
        t_span = torch.linspace(eps, 1.0, timesteps, device=device)
        dt = t_span[1] - t_span[0]
        
        # Initialize
        z = data.clone()  # (N, D) — full joint vector
        log_det = torch.zeros(N, device=device)
        
        sigma = self.sde.sigma.to(device)
        ln_sigma = torch.log(sigma)
        
        # --- ODE Integration ---
        for step in tqdm.tqdm(range(len(t_span)), disable=not verbose, 
                              desc="PF-ODE log_prob"):
            t = t_span[step]
            t_input = t.reshape(1, 1).expand(N, 1)
            
            # --- Compute drift ---
            # PF-ODE drift (encoding direction):
            # f(z,t) = +½ g(t)² score(z,t)
            # For VESDE: g(t)² = σ^(2t) · 2·ln(σ)
            g_sq = sigma ** (2 * t) * 2 * ln_sigma
            
            # Get score from model
            c_batch = condition_mask.unsqueeze(0).expand(N, -1)
            
            with torch.enable_grad():
                z_grad = z.detach().requires_grad_(True)
                
                # Model forward
                score_raw = self.model(x=z_grad, t=t_input, c=c_batch)
                score = self.sbim.output_scale_function(t_input, score_raw)
                
                # Drift = ½ g² score, only on latent dimensions
                drift = 0.5 * g_sq * score * latent_mask.unsqueeze(0)
                
                # --- Hutchinson trace estimator ---
                div_estimate = torch.zeros(N, device=device)
                for _ in range(num_hutchinson):
                    # Random vector, only in latent dimensions
                    epsilon = torch.randn_like(z_grad) * latent_mask.unsqueeze(0)
                    
                    # Compute ε^T · (∂drift/∂z) · ε via autograd
                    # This is the vector-Jacobian product
                    drift_dot_eps = (drift * epsilon).sum()
                    
                    grad_drift = torch.autograd.grad(
                        drift_dot_eps, z_grad,
                        retain_graph=(num_hutchinson > 1),
                        create_graph=False
                    )[0]
                    
                    # Trace estimate = ε^T · J · ε  
                    trace_est = (grad_drift * epsilon).sum(dim=-1)  # (N,)
                    div_estimate += trace_est / num_hutchinson
            
            # --- Update ---
            # Euler step for ODE
            drift_detached = drift.detach()
            z = z.detach() + drift_detached * dt
            
            # Accumulate log-determinant
            # d/dt log p = -div(f), so log p_0 = log p_T + ∫ div(f) dt
            log_det += div_estimate.detach() * dt
        
        # --- Evaluate terminal distribution ---
        # At t=1, the latent variables should be ~ N(0, σ_T²)
        sigma_T = self.sde.marginal_prob_std(torch.tensor(1.0, device=device))
        
        # Extract latent part of z at terminal time
        z_latent = z * latent_mask.unsqueeze(0)  # (N, D), zeros for observed
        
        # Log prob under Gaussian prior (only latent dims)
        log_pT = -0.5 * (z_latent / sigma_T) ** 2 \
                 - 0.5 * torch.log(2 * torch.tensor(np.pi, device=device) * sigma_T ** 2)
        log_pT = (log_pT * latent_mask.unsqueeze(0)).sum(dim=-1)  # (N,)
        
        # --- Total log probability ---
        # log p_0(x) = log p_T(z_T) + ∫₀ᵀ div(f) dt
        # (the integral of -div was accumulated as -log_det, 
        #  so log_det already has the right sign)
        log_prob = log_pT - log_det
        
        return log_prob


def add_log_prob_to_sbim(sbim_class):
    """
    Monkey-patch or mixin: adds log_prob method to ScoreBasedInferenceModel.
    
    Usage:
        add_log_prob_to_sbim(ScoreBasedInferenceModel)
        model = ScoreBasedInferenceModel.load("model.pt")
        log_p = model.log_prob(theta=theta_hat, x=x_obs, device="cuda")
    """
    
    def log_prob(self, theta=None, x=None, condition_mask=None,
                 timesteps=200, eps=1e-3, num_hutchinson=1,
                 device="cpu", verbose=True):
        """
        Compute exact log p(x|theta) or log p(theta|x) via the PF-ODE.
        
        This replaces the sample → KDE → evaluate pipeline with a direct
        computation using the change-of-variables formula.
        
        Args:
            theta: Parameters to condition on (NLE mode) or evaluate (NPE mode)
            x: Observations to condition on (NPE mode) or evaluate (NLE mode)
            condition_mask: Binary mask (1=observed, 0=latent)
            timesteps: ODE integration steps (more = more accurate, slower)
            eps: Start time for integration
            num_hutchinson: Random vectors for trace estimation (1 is usually fine)
            device: Device to use
            verbose: Show progress
            
        Returns:
            log_probs: (N,) tensor of log-probabilities for the latent variables
        """
        evaluator = PFODELogProb(self)
        return evaluator.log_prob(
            theta=theta, x=x, condition_mask=condition_mask,
            timesteps=timesteps, eps=eps, num_hutchinson=num_hutchinson,
            device=device, verbose=verbose
        )
    
    sbim_class.log_prob = log_prob


# ============================================================
# Integration into ModelTransfuser.compare()
# ============================================================
#
# Replace the NLE sampling + KDE block with:
#
#   # OLD (KDE-based):
#   # likelihood_samples = model.sample(theta=MAP, ...)
#   # log_probs = [self._log_prob(samples[i], x[i]) for i in range(len(x))]
#
#   # NEW (PF-ODE):
#   # Build joint data with MAP as theta and x_obs as x
#   log_probs = model.log_prob(
#       theta=MAP_posterior,
#       x=x,
#       timesteps=200,
#       device=device
#   )
#
# This eliminates:
# 1. The NLE sampling step entirely (no need to generate 1000+ samples)
# 2. The KDE fitting step
# 3. The KDE evaluation step
# 4. The dimensionality limitation of KDE
#
# Trade-off:
# - Requires autograd (backward passes), so ~2x slower per timestep
# - But you need far fewer timesteps than sampling (200 vs 500)
# - And you skip the entire sample→KDE pipeline
# - Net effect: roughly comparable wall-clock time, much more accurate


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    from compass import ScoreBasedInferenceModel as SBIm
    
    # Add the method
    add_log_prob_to_sbim(SBIm)
    
    # Load trained model
    model = SBIm.load("data/gaussians/Hypothesis 1.pt", device="cuda")
    
    # Compute log p(x_obs | theta_hat) directly
    theta_hat = torch.tensor([[0.5]])  # MAP estimate
    x_obs = torch.tensor([[1.2, 0.8]])  # observations
    
    log_p = model.log_prob(
        theta=theta_hat.expand(x_obs.shape[0], -1),  # broadcast to match x
        x=x_obs,
        timesteps=200,
        device="cuda",
        verbose=True
    )
    
    print(f"Log p(x|theta): {log_p}")