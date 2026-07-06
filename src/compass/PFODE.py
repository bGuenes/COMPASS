import torch
import tqdm

#################################################################################################
# ////////////////////////// Probability-Flow ODE log-probability ///////////////////////////////
#################################################################################################

class PFODE():
    """
    Exact(-in-the-limit) log-probability evaluation through the probability-flow ODE
    (Song et al. 2021, "Score-Based Generative Modeling through SDEs", Sec. 4.3;
    instantaneous change of variables: Chen et al. 2018, Neural ODEs).

    The score network s(x, t) defines a continuous normalizing flow. Integrating the
    probability-flow ODE from the data at t = eps forward to t = 1 while accumulating
    the score divergence gives the log-density of the (eps-smoothed) model distribution:

        log p_eps(x) = log p_1(x(1)) - int_{lambda_min}^{lambda_max} lambda * div s_y dlambda
                       - d * log alpha(eps)

    written in the rescaled variables y = x / alpha(t) and noise-to-signal scale
    lambda(t) = sigma(t) / alpha(t), in which every supported SDE is variance
    exploding (dy/dlambda = -lambda * s_y). For the VESDE alpha = 1, lambda = sigma.
    p_1 is approximated by the SDE's Gaussian prior N(0, lambda_max^2) on the
    rescaled state (standard practice; exact up to Var(data)/lambda_max^2).

    Because the network is a *conditional* score model (condition mask c), this
    evaluates the conditional density of the latent dimensions given the observed
    ones: with theta conditioned it returns log p(x | theta) (the likelihood),
    with x conditioned it returns log p(theta | x) (the posterior).

    Unlike a KDE over model samples, this
      - has no bandwidth/smoothing bias (smoothing is only sigma(eps) ~ 0.03),
      - does not degrade with dimension (no curse-of-dimensionality of KDEs),
      - is deterministic given the trained network (no sampling noise).
    """

    def __init__(self, SBIm):
        self.SBIm = SBIm
        self.sde = SBIm.sde

    #############################################
    # ----- Log-probability -----
    #############################################

    @torch.no_grad()
    def log_prob(self, data, condition_mask, timesteps=100, eps=1e-3,
                 divergence="exact", hutchinson_samples=32,
                 device="cpu", batch_size=4096, verbose=False):
        """
        Log-probability of the latent dimensions of `data` given its conditioned
        dimensions, evaluated with the probability-flow ODE.

        Args:
            data:           Joint node vectors, shape (num_points, nodes_size).
                            Conditioned dimensions hold the conditioning values,
                            latent dimensions hold the point the density is
                            evaluated at.
            condition_mask: Binary mask, 1 = conditioned, 0 = latent (the
                            dimensions whose density is evaluated).
                            Shape (nodes_size,) or (num_points, nodes_size).
            timesteps:      Number of integration nodes of the Heun (2nd order)
                            solver on the log-lambda grid.
            eps:            Diffusion end time; the returned density is the model
                            density smoothed with N(0, sigma(eps)^2).
            divergence:     "exact" (default) computes the exact divergence with
                            one backward pass per latent dimension -- preferred
                            for the low-dimensional problems COMPASS targets.
                            "hutchinson" uses a Rademacher trace estimator.
            hutchinson_samples: Number of probe vectors for "hutchinson".
            device:         Device to run on.
            batch_size:     Points per integration batch.
            verbose:        Show a progress bar over integration steps.

        Returns:
            log_prob: tensor of shape (num_points,) on CPU.
        """
        model = self.SBIm.model.to(device)
        was_training = model.training
        model.eval()

        data = torch.as_tensor(data, dtype=torch.float32)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        condition_mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        if condition_mask.dim() == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        log_probs = []
        for start in range(0, data.shape[0], int(batch_size)):
            batch = data[start:start + int(batch_size)].to(device)
            mask = condition_mask[start:start + int(batch_size)].to(device)
            log_probs.append(self._log_prob_batch(batch, mask, timesteps, eps,
                                                  divergence, hutchinson_samples,
                                                  device, verbose))
        if was_training:
            model.train()
        return torch.cat(log_probs, dim=0).cpu()

    def _log_prob_batch(self, data, condition_mask, timesteps, eps,
                        divergence, hutchinson_samples, device, verbose):
        sde = self.sde
        latent = (1 - condition_mask)

        one = torch.ones(1, device=device)
        lam_min = sde.lambda_t(eps * one).item()
        lam_max = sde.lambda_t(one).item()
        lams = torch.logspace(torch.log10(torch.tensor(lam_min)),
                              torch.log10(torch.tensor(lam_max)),
                              timesteps, device=device)
        ts = sde.time_of_lambda(lams)

        # Rescaled state y = x / alpha(t) on the latent dims; conditioned dims
        # enter the network unscaled (they are never diffused during training).
        alpha_eps = sde.alpha_t(ts[0]).to(device)
        y = data.clone()
        y = y * (latent / alpha_eps + condition_mask)

        # Forward integration (data -> noise) with Heun's method, accumulating
        # the divergence integral I = int lambda * div_y(s_y) dlambda.
        I = torch.zeros(data.shape[0], device=device)
        for i in tqdm.tqdm(range(timesteps - 1), disable=not verbose,
                           desc="PF-ODE log-prob"):
            lam0, lam1 = lams[i], lams[i + 1]
            h = lam1 - lam0

            s0, div0 = self._score_and_div(y, ts[i], condition_mask, latent,
                                           divergence, hutchinson_samples)
            y_pred = y + h * (-lam0 * s0) * latent
            s1, div1 = self._score_and_div(y_pred, ts[i + 1], condition_mask, latent,
                                           divergence, hutchinson_samples)

            y = y + 0.5 * h * (-lam0 * s0 - lam1 * s1) * latent
            I = I + 0.5 * h * (lam0 * div0 + lam1 * div1)

        # Gaussian prior of the SDE at t=1 in the rescaled variables,
        # p_1(y) ~ N(mu_1, lambda_max^2). Instead of assuming mu_1 = 0 (which
        # biases the result whenever the data is not zero-centered, since the
        # VESDE preserves the data mean), the mean is eliminated with Tweedie's
        # formula at the endpoint: s_1(y) = -(y - mu_1)/lambda_max^2 up to
        # O(Var(data)/lambda_max^2), hence (y - mu_1)^2 / (2 lambda_max^2)
        # = lambda_max^2 * s_1(y)^2 / 2. This makes the evaluation independent
        # of the data location, so no normalization of the data is required.
        s_end, _ = self._score_and_div(y, ts[-1], condition_mask, latent,
                                       divergence, hutchinson_samples,
                                       compute_div=False)
        d = latent.sum(dim=-1)
        log_prior = (-0.5 * torch.log(2 * torch.pi * lams[-1]**2) * d
                     - 0.5 * lams[-1]**2 * (s_end**2 * latent).sum(dim=-1))

        # Change of variables x = alpha * y at the data end.
        return log_prior - I - d * torch.log(alpha_eps)

    def _score_and_div(self, y, t, condition_mask, latent, divergence, hutchinson_samples,
                       compute_div=True):
        """Score s_y = alpha * s_x and its divergence w.r.t. y over the latent dims.

        The network approximates s_x = model(x,t,c)/sigma(t) with x = alpha(t) y on
        the latent dims; div_y(s_y) = alpha^2 * div_x(s_x).
        """
        sde = self.sde
        alpha = sde.alpha_t(t).to(y.device)
        sigma = sde.sigma_t(t).to(y.device)
        t_vec = (t * torch.ones(y.shape[0], 1, device=y.device))

        x = y * (alpha * latent + condition_mask)

        if not compute_div:
            with torch.no_grad():
                out = self.SBIm.model(x=x, t=t_vec, c=condition_mask)
                s_x = out / sigma
            return (alpha * s_x).detach(), None

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            out = self.SBIm.model(x=x, t=t_vec, c=condition_mask)
            s_x = out / sigma

            if divergence == "exact":
                # One backward pass per latent dimension: exact and deterministic,
                # cheap for the low-dimensional joints COMPASS works with.
                latent_dims = torch.nonzero(latent.any(dim=0)).flatten().tolist()
                div = torch.zeros(y.shape[0], device=y.device)
                for k, j in enumerate(latent_dims):
                    grad_j = torch.autograd.grad(s_x[:, j].sum(), x,
                                                 retain_graph=(k < len(latent_dims) - 1))[0]
                    div = div + grad_j[:, j] * latent[:, j]
            elif divergence == "hutchinson":
                div = torch.zeros(y.shape[0], device=y.device)
                for k in range(hutchinson_samples):
                    v = (torch.randint(0, 2, y.shape, device=y.device).float() * 2 - 1) * latent
                    grad_v = torch.autograd.grad((s_x * v).sum(), x,
                                                 retain_graph=(k < hutchinson_samples - 1))[0]
                    div = div + (grad_v * v).sum(dim=-1)
                div = div / hutchinson_samples
            else:
                raise ValueError(f"Unknown divergence method '{divergence}'")

        s_y = (alpha * s_x).detach()
        div_y = (alpha**2 * div).detach()
        return s_y, div_y

    #############################################
    # ----- Score-ascent MAP -----
    #############################################

    @torch.no_grad()
    def map_estimate(self, data, condition_mask, init=None,
                     sigma_start=None, timesteps=100, eps=1e-3,
                     iterations_per_level=3, device="cpu"):
        """
        KDE-free MAP estimate of the latent dimensions by deterministic annealed
        score ascent (mean-shift with the learned score).

        Repeatedly applying the Tweedie denoising update
            z <- z + sigma_k^2 * s(z, sigma_k)
        at a fixed noise level sigma_k converges to a local mode of the
        sigma_k-smoothed density; annealing sigma_k from ~ the posterior scale
        down to sigma(eps) tracks that mode to (almost) the unsmoothed MAP.
        Compared to a KDE mode this has no bandwidth bias: a KDE both broadens the
        distribution by its bandwidth and, in more than a couple of dimensions,
        needs far more samples than are available for a stable mode.

        Args:
            data:           Joint node vectors (num_points, nodes_size); conditioned
                            dims hold the conditioning values, latent dims the
                            starting point (e.g. posterior sample mean) unless
                            `init` is given.
            condition_mask: 1 = conditioned, 0 = latent (the dims optimized over).
            init:           Optional starting values for the latent dims
                            (num_points, nodes_size), defaults to `data`.
            sigma_start:    Starting noise scale of the annealing; defaults to the
                            per-point std of the latent init spread if `init` is
                            2D over samples, else 1.0.
            timesteps:      Number of annealing levels (geometric in lambda).
            eps:            Final diffusion time.
            iterations_per_level: Fixed-point iterations per noise level.
            device:         Device to run on.

        Returns:
            Tensor (num_points, nodes_size) with the latent dims replaced by the
            MAP estimate.
        """
        sde = self.sde
        model = self.SBIm.model.to(device)
        model.eval()

        data = torch.as_tensor(data, dtype=torch.float32).clone()
        if data.dim() == 1:
            data = data.unsqueeze(0)
        condition_mask = torch.as_tensor(condition_mask, dtype=torch.float32)
        if condition_mask.dim() == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        z = (data if init is None else torch.as_tensor(init, dtype=torch.float32).clone())
        z = z.to(device)
        mask = condition_mask.to(device)
        latent = 1 - mask

        one = torch.ones(1, device=device)
        lam_min = sde.lambda_t(eps * one).item()
        lam_max = sde.lambda_t(one).item()
        if sigma_start is None:
            sigma_start = 1.0
        # Keep the annealing inside the trained noise range [lambda(eps), lambda(1)]
        lam_hi = min(max(float(sigma_start), 2 * lam_min), lam_max)
        lams = torch.logspace(torch.log10(torch.tensor(lam_hi)),
                              torch.log10(torch.tensor(lam_min)),
                              timesteps, device=device)
        ts = sde.time_of_lambda(lams)

        for i in range(timesteps):
            t = ts[i]
            alpha = sde.alpha_t(t).to(device)
            sigma = sde.sigma_t(t).to(device)
            lam = lams[i]
            t_vec = t * torch.ones(z.shape[0], 1, device=device)
            for _ in range(iterations_per_level):
                x = z * (alpha * latent + mask)
                s_x = model(x=x, t=t_vec, c=mask) / sigma
                # Tweedie denoiser in the rescaled variables: y + lam^2 * s_y
                z = z + lam**2 * (alpha * s_x) * latent

        # Back to x-space (z is the rescaled state y on the latent dims).
        alpha_final = sde.alpha_t(ts[-1]).to(device)
        z = z * (alpha_final * latent + mask)
        return z.cpu()
