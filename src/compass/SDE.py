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

#############################################
# ----- VPSDE -----
#############################################
class VPSDE():
    def __init__(self):
        raise NotImplementedError("VPSDE is not implemented yet.")
    