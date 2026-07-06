import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import tqdm
import datetime
import os

class TensorTupleDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        assert len(tensor1) == len(tensor2), "Tensors must have the same length"

    def __len__(self):
        return len(self.tensor1)

    def __getitem__(self, idx):
        return self.tensor1[idx], self.tensor2[idx], idx

#################################################################################################
# ///////////////////////////////// Multi-Observation Sampling //////////////////////////////////
#################################################################################################
class MultiObsSampler():
    """
    Compositional score modeling for inference with multiple i.i.d. observations
    (F-NPSE, Geffner et al. 2023, https://arxiv.org/abs/2209.14249).

    The posterior given n observations factorizes over the single-observation posteriors:

        p(theta | x_1, ..., x_n)  ∝  p(theta)^(1-n) * prod_j p(theta | x_j)

    so a score network trained on single (theta, x) pairs can be composed at inference
    time. The `hierarchy` indices select the *shared* (global) parameters that are
    composed across observations; latent dimensions not in `hierarchy` are treated as
    per-observation (local) latents and keep their individual scores.

    Three composition rules are implemented (`correction` argument):

    - "gauss" (default): Gaussian-corrected composition (Gloeckler et al. 2024,
      "Compositional simulation-based inference for time series"). The composed score
      approximates the score of the *diffused* multi-observation posterior, so it is
      valid inside the standard reverse-diffusion samplers (euler / dpm). Requires an
      estimate of the single-observation posterior precision on the hierarchy
      dimensions, which is estimated automatically if not provided.
    - "uncorrected": (1-n) * diffused-prior score + sum of individual scores.
      Cheap, exact as t -> 0, but biased at large t.
    - "fnpe": the exact Eq. 7 of Geffner et al.: (1-n)(1-t) * prior score + sum of
      individual scores. This is the score of the paper's bridging densities, which are
      NOT the diffusion marginals of the posterior: it is only consistent with annealed
      Langevin sampling (use method="langevin"). The initial noise and Langevin step
      sizes on the hierarchy dimensions are scaled by 1/n internally (reference
      N(0, sigma_max^2 / n)).

    Practical notes:
    - Hierarchical models (per-observation local latents alongside the shared
      parameters, e.g. hierarchy=[0, 1] with further latent dimensions) benefit from
      dense Langevin correction: corrector_steps_interval=1, corrector_steps=10,
      snr=0.2. With the default settings the posterior width can come out conservative
      (up to ~1.5x too wide at n ~ 200) because the diagonal Gaussian correction
      ignores the global-local coupling; posterior means are unaffected.
    - For large n the composed posterior contracts like 1/sqrt(n), so its quality is
      limited by the score network's systematic error: a network bias of delta in
      parameter units stays a bias of ~delta, which becomes more visible relative to
      the shrinking posterior width. Training with more simulations/epochs (and the
      default "mixture" time sampling of the Trainer) directly improves this.
    """

    def __init__(self, SBIm):
        self.SBIm = SBIm
        # Get SDE from model for calculations
        self.sde = self.SBIm.sde

    #############################################
    # ----- Main Sampling Loop -----
    #############################################

    def sample(self, world_size, data, condition_mask=None, timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, hierarchy=None,
               prior=None, correction="gauss", posterior_precision=None,
               precision_est_samples=500, precision_est_timesteps=None, denoise_clamp=5.0,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cpu", verbose=True, method="dpm", save_trajectory=False, result_dict=None):
        """
        Sample from the multi-observation posterior via compositional score modeling.

        Args:
            data: Input data
                    - Tensor of observed values, one row per observation
                        - Shape data: (num_observations, num_observed_features)
                    - condition_mask must be provided (or inferable)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_total_features,) or (num_observations, num_total_features)
            timesteps: Number of diffusion steps
            eps: End time for diffusion process
            num_samples: Number of samples to generate
            cfg_alpha: Classifier-free guidance strength
            hierarchy: Indices of the *shared* (global) latent variables that are composed
                    across observations. Defaults to all latent variables.
            prior: Gaussian prior over the hierarchy dimensions, as a tuple
                    (mean, std) of tensors/lists/floats of length len(hierarchy).
                    Defaults to a standard normal N(0, 1) (correct if the model was
                    trained on parameters standardized to zero mean and unit variance).
            correction: Composition rule: "gauss" (default), "uncorrected" or "fnpe".
            posterior_precision: Optional estimate of the single-observation posterior
                    precision on the hierarchy dimensions, used by the "gauss" correction.
                    Shape (len(hierarchy),) or (num_observations, len(hierarchy)).
                    If None, it is estimated by sampling the single-observation
                    posteriors once with the standard sampler.
            precision_est_samples: Number of samples per observation for the automatic
                    precision estimation.
            precision_est_timesteps: Number of diffusion steps for the automatic
                    precision estimation (defaults to `timesteps`).
            denoise_clamp: Clamp the denoised predictions theta_t + sigma_t^2 * score
                    to within this many prior standard deviations of the prior mean
                    (stabilizes the score composition in the tails of the reference
                    distribution). None disables clamping.

            - DPM-Solver parameters -
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end

            - Other parameters -
            device: Device to run sampling on
            verbose: Whether to show progress bar
            method: Sampling method to use (euler, dpm)
            save_trajectory: Whether to save the intermediate denoising trajectory
        """

        # Set parameters
        self.world_size = world_size
        self.timesteps = timesteps
        self.eps = eps
        self.num_samples = int(num_samples)
        self.cfg_alpha = cfg_alpha
        self.verbose = verbose
        self.method = method
        self.save_trajectory = save_trajectory
        self.hierarchy = hierarchy
        self.correction = correction
        self.denoise_clamp = denoise_clamp

        if method in ("dpm", "langevin"):
            self.corrector_steps_interval = corrector_steps_interval
            self.corrector_steps = corrector_steps
            self.final_corrector_steps = final_corrector_steps
            self.snr = snr
            self.order = order

        if correction not in ("gauss", "uncorrected", "fnpe"):
            raise ValueError(f"Unknown correction '{correction}'. Choose from 'gauss', 'uncorrected', 'fnpe'.")
        if correction == "fnpe" and method != "langevin":
            print("WARNING: correction='fnpe' composes the scores of the F-NPSE bridging densities, "
                  "which are NOT the diffusion marginals of the posterior. Reverse-diffusion samplers "
                  "(euler/dpm) diverge with it; use method='langevin' (annealed Langevin dynamics).")

        # Resolve hierarchy before spawning workers (needed for prior & precision setup)
        if self.hierarchy is None:
            if condition_mask is None:
                raise ValueError("Either hierarchy or condition_mask must be provided.")
            cm = condition_mask if condition_mask.dim() == 1 else condition_mask[0]
            self.hierarchy = torch.where(cm == 0)[0].tolist()

        # Resolve the Gaussian prior over the hierarchy dimensions
        self.prior_mean, self.prior_std = self._resolve_prior(prior)

        # Posterior precision estimates for the Gaussian correction
        if correction == "gauss":
            if posterior_precision is None:
                if verbose:
                    print("Estimating single-observation posterior precisions for the 'gauss' correction ...")
                posterior_precision = self._estimate_posterior_precision(
                    data, condition_mask,
                    num_samples=precision_est_samples,
                    timesteps=precision_est_timesteps or timesteps,
                    eps=eps, device=device if world_size <= 1 else "cuda:0")
            self.posterior_precision = self._validate_precision(torch.as_tensor(posterior_precision, dtype=torch.float32))
        else:
            self.posterior_precision = None

        if self.world_size > 1:
            manager = mp.Manager()
            result_dict = manager.dict()
            mp.spawn(self._sample_loop, args=(data, condition_mask, num_samples, result_dict), nprocs=self.world_size, join=True)
            samples = result_dict.get('samples', None)
            manager.shutdown()

        else:
            rank = 0
            self.device = device
            samples = self._sample_loop(rank, data, condition_mask, num_samples)

        return samples

    def _sample_loop(self, rank, data, condition_mask, num_samples, result_dict=None):
        # Set rank
        self.rank = rank
        self.verbose = self.verbose if self.rank == 0 else False

        # Set distributed parameters
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)

        # Check data structure
        data_loader, self.num_observations = self._check_data_structure(data, condition_mask)

        # Move prior (and precision estimates) to device
        self.prior_mean = self.prior_mean.to(self.device)
        self.prior_std = self.prior_std.to(self.device)
        if self.posterior_precision is not None:
            self.posterior_precision = self.posterior_precision.to(self.device)

        # Set up timesteps on a geometric noise-scale grid (log-spaced sigma), which
        # resolves the small-noise end far better than a uniform time grid. The grid
        # is NOT extended below sigma_m(eps): the score network is unreliable below
        # the noise scales it was trained on. With the "gauss" correction the reverse
        # diffusion additionally stops once the noise scale reaches the width of the
        # composed posterior (known from the precision estimates): below that scale
        # the score network cannot add information, and the remaining gap to t=0 is
        # closed exactly (under the Gaussian approximation) by the final analytic
        # denoising step (see _final_denoise).
        one = torch.ones(1, device=self.device)
        sigma_max = self.sde.marginal_prob_std(one)
        sigma_min = self.sde.marginal_prob_std(self.eps * one)
        if self.correction == "gauss":
            n = self.num_observations
            Lambda_prior = 1.0 / self.prior_std**2
            if self.posterior_precision.shape[0] == 1:
                Lambda_star = (1 - n) * Lambda_prior + n * self.posterior_precision[0]
            else:
                Lambda_star = (1 - n) * Lambda_prior + self.posterior_precision.sum(dim=0)
            sigma_stop = (1.0 / Lambda_star).sqrt().min()
            sigma_min = torch.maximum(sigma_min, sigma_stop.reshape(1).to(self.device))
        sigmas = torch.logspace(torch.log10(sigma_max).item(), torch.log10(sigma_min).item(),
                                self.timesteps, device=self.device)
        self.timesteps_list = self.sde.time_of_sigma(sigmas)

        # Loop over data samples
        all_samples = []
        indices = []
        for batch in data_loader:
            # Prepare data for sampling
            data_batch, condition_mask_batch, idx = self._prepare_data(batch, num_samples, self.device)

            # Draw samples from initial noise distribution
            data_batch = self._initial_sample(data_batch, condition_mask_batch)

            # Get samples for this batch
            if self.method == "euler":
                samples = self._basic_sampler(data_batch, condition_mask_batch, idx)
            elif self.method == "dpm":
                samples = self._dpm_sampler(data_batch, condition_mask_batch, idx,
                                            order=self.order, snr=self.snr, corrector_steps_interval=self.corrector_steps_interval,
                                            corrector_steps=self.corrector_steps, final_corrector_steps=self.final_corrector_steps)
            elif self.method == "langevin":
                samples = self._langevin_sampler(data_batch, condition_mask_batch, idx,
                                                 snr=self.snr, steps_per_level=self.corrector_steps)
            else:
                raise ValueError(f"Sampling method {self.method} not recognized.")

            # Final analytic denoising step (gauss correction only)
            if self.correction == "gauss":
                samples = self._final_denoise(samples, condition_mask_batch, idx)

            # Store samples
            all_samples.append(samples)
            indices.append(idx)

        # Collect results from all processes if distributed
        if self.world_size > 1:
            dist.barrier()
            self._gather_samples(all_samples, indices, result_dict)
            dist.barrier()
            dist.destroy_process_group()

        else:
            samples = torch.cat(all_samples, dim=0)
            return samples

    #############################################
    # ----- Prior & Precision Setup -----
    #############################################

    def _resolve_prior(self, prior):
        """Resolve the Gaussian prior over the hierarchy dimensions to (mean, std) tensors."""
        n_hierarchy = len(self.hierarchy)
        if prior is None:
            print("WARNING: No prior provided for compositional score modeling. "
                  "Assuming a standard normal prior N(0, 1) over the shared parameters. "
                  "Pass prior=(mean, std) if this does not match your model.")
            return torch.zeros(n_hierarchy), torch.ones(n_hierarchy)

        mean, std = prior
        mean = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(std, dtype=torch.float32).flatten()
        if mean.numel() == 1:
            mean = mean.repeat(n_hierarchy)
        if std.numel() == 1:
            std = std.repeat(n_hierarchy)
        if mean.numel() != n_hierarchy or std.numel() != n_hierarchy:
            raise ValueError(f"Prior mean/std must have length {n_hierarchy} (= len(hierarchy)), "
                             f"got {mean.numel()}/{std.numel()}.")
        return mean, std

    def _estimate_posterior_precision(self, data, condition_mask, num_samples, timesteps, eps, device):
        """
        Estimate the precision of each single-observation posterior on the hierarchy
        dimensions by sampling p(theta | x_j) once with the standard (single-observation)
        sampler and moment-matching a Gaussian.
        """
        samples = self.SBIm.sampler.sample(
            world_size=1, data=data, condition_mask=condition_mask,
            timesteps=timesteps, eps=eps, num_samples=num_samples,
            device=device, verbose=self.verbose, method="dpm")

        # samples: (num_observations, num_samples, num_features)
        var = samples[:, :, self.hierarchy].var(dim=1)
        precision = 1.0 / torch.clamp(var, min=1e-8)
        return precision.cpu()

    def _validate_precision(self, precision):
        """
        Bring the precision estimates to shape (num_hierarchy,) or (n_obs, num_hierarchy) and
        make sure the composed precision (1-n)*Lambda_prior + sum_j Lambda_j stays positive
        (Gloeckler et al. 2024). Where it does not, the deficit is distributed over the
        individual precisions.
        """
        precision = torch.atleast_2d(precision.float())  # (n_obs or 1, H)
        n = precision.shape[0]
        if n > 1:
            prior_precision = 1.0 / self.prior_std**2
            composed = (1 - n) * prior_precision + precision.sum(dim=0)
            deficit = torch.clamp(-composed, min=0.0)
            nudge = deficit / (n - 1) + torch.where(deficit > 0, torch.full_like(deficit, 0.1), torch.zeros_like(deficit))
            precision = precision + nudge
        return precision

    #############################################
    # ----- Multi-GPU setup -----
    #############################################

    def _ddp_setup(self, rank, world_size):
        # Setup DistributedDataParallel
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ["MASTER_PORT"] = "29500"

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=100_000_000)
        )

        self.SBIm.model.eval()
        self.device = torch.device(f'cuda:{rank}')
        self.SBIm.model.to(self.device)
        self.model = DDP(self.SBIm.model, device_ids=[rank])

    def _gather_samples(self, all_samples, indices, result_dict):
        # Gather samples from all processes
        # Convert the list of tensors to a single tensor
        samples = torch.cat(all_samples, dim=0).to(self.device)
        indices = torch.cat(indices, dim=0).to(self.device)

        # Create empty tensors to gather results across processes
        gathered_samples = [torch.zeros_like(samples) for _ in range(self.world_size)]
        gathered_idx = [torch.zeros_like(indices) for _ in range(self.world_size)]

        # Gather data from all processes
        dist.all_gather(gathered_samples, samples)
        dist.all_gather(gathered_idx, indices)

        if self.rank == 0:
            # Sort results by index
            gathered_idx = torch.cat(gathered_idx, dim=0)
            gathered_samples = torch.cat(gathered_samples, dim=0)
            unique_sort_idx = [(gathered_idx == i).nonzero()[0,0].tolist() for i in gathered_idx.unique()]
            samples = gathered_samples[unique_sort_idx]

            result_dict['samples'] = samples.cpu()

    def _gather_scores(self, score_table, indices):
        # Gather scores from all processes and deduplicate
        # (the DistributedSampler pads shards, so observations can appear twice)
        gathered_scores = [torch.zeros_like(score_table) for _ in range(self.world_size)]

        indices = indices.to(self.device)
        gathered_idx = [torch.zeros_like(indices) for _ in range(self.world_size)]

        dist.all_gather(gathered_scores, score_table)
        dist.all_gather(gathered_idx, indices)

        # Sort results by index, keeping one entry per unique observation
        gathered_idx = torch.cat(gathered_idx, dim=0)
        gathered_scores = torch.cat(gathered_scores, dim=0)
        unique_sort_idx = [(gathered_idx == i).nonzero()[0,0].tolist() for i in gathered_idx.unique()]
        gathered_scores = gathered_scores[unique_sort_idx]

        return gathered_scores

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask, indices, cfg_alpha=None):
        """Get the composed score estimate with optional classifier-free guidance"""
        with torch.no_grad():
            n_obs, num_samples, num_features = x.shape

            # Flatten (n_obs, num_samples, num_features) -> (n_obs*num_samples, num_features)
            # so all observations are processed in a single forward pass
            x_flat = x.reshape(n_obs * num_samples, num_features)
            c_flat = condition_mask.reshape(n_obs * num_samples, num_features)

            scores_flat = self.SBIm.model(x=x_flat, t=t, c=c_flat)
            scores_flat = self.SBIm.output_scale_function(t, scores_flat)

            if cfg_alpha is not None:
                scores_uncond = self.SBIm.model(x=x_flat, t=t, c=torch.zeros_like(c_flat))
                scores_uncond = self.SBIm.output_scale_function(t, scores_uncond)
                scores_flat = scores_uncond + cfg_alpha * (scores_flat - scores_uncond)

            score_table = scores_flat.reshape(n_obs, num_samples, num_features)

            if self.world_size > 1:
                dist.barrier()
                score_table = self._gather_scores(score_table, indices)

            score = self._compositional_score(score_table, x, t)

        if self.world_size > 1:
            return score[indices]
        return score

    def _compositional_score(self, scores, x, t):
        """
        Compose the per-observation scores on the hierarchy (shared parameter)
        dimensions. Local latent dimensions keep their per-observation score.

        The state x is kept synchronized across the observation axis on the hierarchy
        dimensions, so x[0] provides the shared parameter values.

        - "fnpe" (Geffner et al. 2023, Eq. 7):
              s = (1-n)(1-t) * prior_score + sum_j s_j
          with the undiffused prior score, valid for annealed Langevin sampling only.
        - "uncorrected":
              s = (1-n) * diffused_prior_score + sum_j s_j
        - "gauss" (Gloeckler et al. 2024):
              s = Lambda^-1 [ (1-n) * Lambda_prior * diffused_prior_score
                              + sum_j Lambda_j * s_j ]
          where Lambda_prior / Lambda_j are the *denoising* precisions of prior and
          single-observation posteriors, and Lambda = (1-n) Lambda_prior + sum_j Lambda_j.
        """
        h = self.hierarchy
        n = scores.shape[0]

        var_t = self.sde.marginal_prob_std(t).to(x.device)**2   # (1, 1)

        theta_h = x[:1, :, h]                                   # (1, num_samples, H), shared

        # Stabilize the individual scores by clamping their denoised prediction
        # E[theta_0 | theta_t, x_j] = theta_t + var_t * s_j to a box around the prior.
        # Far out in the tails of the reference distribution the network
        # extrapolates; without this, single bad tail scores get amplified n-fold
        # by the composition and samples can run away.
        if self.denoise_clamp is not None:
            lo = self.prior_mean - self.denoise_clamp * self.prior_std
            hi = self.prior_mean + self.denoise_clamp * self.prior_std
            x0 = torch.clamp(theta_h + var_t * scores[:, :, h], min=lo, max=hi)
            scores[:, :, h] = (x0 - theta_h) / var_t

        sum_scores = scores[:, :, h].sum(dim=0, keepdim=True)   # (1, num_samples, H)

        if self.correction == "fnpe":
            # Eq. 7 with the *diffused* prior score. With the undiffused prior of the
            # paper (which assumes a VP-type diffusion and a standard-normal prior),
            # the VESDE bridging densities become improper for large n at
            # intermediate t (negative total precision) and Langevin diverges.
            # Diffusing the prior keeps them proper and recovers Eq. 7 as t -> 0.
            prior_score = -(theta_h - self.prior_mean) / (self.prior_std**2 + var_t)
            composed = (1 - n) * (1 - t) * prior_score + sum_scores

        elif self.correction == "uncorrected":
            prior_score = -(theta_h - self.prior_mean) / (self.prior_std**2 + var_t)
            composed = (1 - n) * prior_score + sum_scores

        elif self.correction == "gauss":
            prior_score = -(theta_h - self.prior_mean) / (self.prior_std**2 + var_t)
            Lambda_prior = 1.0 / self.prior_std**2 + 1.0 / var_t                  # (1, H)
            Lambda_j = self.posterior_precision + 1.0 / var_t                     # (n or 1, H)
            if Lambda_j.shape[0] == 1:
                Lambda_sum = n * Lambda_j
                weighted_sum = Lambda_j * sum_scores
            else:
                Lambda_sum = Lambda_j.sum(dim=0, keepdim=True)
                weighted_sum = (Lambda_j.unsqueeze(1) * scores[:, :, h]).sum(dim=0, keepdim=True)
            Lambda = (1 - n) * Lambda_prior + Lambda_sum                          # (1, H)
            composed = ((1 - n) * Lambda_prior * prior_score + weighted_sum) / Lambda

        # Clamp the denoised prediction of the composed score as well. Not valid for
        # "fnpe": its bridging-density score does not have the diffused-posterior
        # form theta + var_t * s = E[theta_0|theta_t] that the clamp assumes.
        if self.denoise_clamp is not None and self.correction != "fnpe":
            x0 = torch.clamp(theta_h + var_t * composed, min=lo, max=hi)
            composed = (x0 - theta_h) / var_t

        # Broadcast the composed score to all observation rows on the shared dims
        scores[:, :, h] = composed

        return scores

    def _final_denoise(self, x, condition_mask, idx):
        """
        Analytic denoising (Tweedie) step from the final diffusion time t_end to t=0
        on the shared (hierarchy) dimensions.

        The reverse diffusion stops at sigma_m(eps), so the samples follow the
        posterior *convolved* with N(0, sigma_m(eps)^2) -- a visible overdispersion
        once the composed posterior gets narrower than sigma_m(eps) (it contracts
        like 1/sqrt(n)). Under the Gaussian (posterior-precision) approximation
        already used by the "gauss" correction, p(theta_0 | theta_t_end) is Gaussian
        with mean theta + sigma^2 * score(theta) and precision Lambda* + 1/sigma^2,
        where Lambda* = (1-n) Lambda_prior + sum_j Lambda_j is the composed posterior
        precision at t=0. Drawing from it removes the leftover noise exactly.
        """
        h = self.hierarchy
        t_end = self.timesteps_list[-1].reshape(-1, 1)
        var_end = self.sde.marginal_prob_std(t_end).to(x.device)**2

        score = self._get_score(x, t_end, condition_mask, idx, self.cfg_alpha)

        n = self.num_observations
        Lambda_prior = 1.0 / self.prior_std**2
        if self.posterior_precision.shape[0] == 1:
            Lambda_star = (1 - n) * Lambda_prior + n * self.posterior_precision[0]
        else:
            Lambda_star = (1 - n) * Lambda_prior + self.posterior_precision.sum(dim=0)
        var_denoise = 1.0 / (Lambda_star + 1.0 / var_end)                     # (1, H)

        noise = self._shared_noise(x)
        x = x.clone()
        x[:, :, h] = x[:, :, h] + var_end * score[:, :, h] \
                     + torch.sqrt(var_denoise) * noise[:, :, h]
        return x

    def _shared_noise(self, x):
        """
        Noise that is shared across the observation axis on the hierarchy dimensions
        (so the shared parameters stay synchronized across observation rows) and
        independent on all other dimensions.
        """
        noise = torch.randn_like(x)
        shared = torch.randn(1, x.shape[1], x.shape[2], device=x.device)
        if self.world_size > 1:
            dist.broadcast(shared, src=0)
        noise[:, :, self.hierarchy] = shared[:, :, self.hierarchy]
        return noise

    def _hierarchy_scale(self, x):
        """
        Per-feature scale for initial noise / Langevin steps. For "fnpe" the bridging
        densities have reference N(0, sigma_max^2 / n) on the shared dimensions
        (Geffner et al. 2023), so noise and step sizes shrink by 1/n there.
        """
        scale = torch.ones(x.shape[2], device=x.device)
        if self.correction == "fnpe":
            scale[self.hierarchy] = 1.0 / self.num_observations
        return scale

    def _check_data_shape(self, data, condition_mask):
        # Check data shape
        # Required shape: (num_samples, num_features)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)

        # Check condition mask shape
        # Required shape: (num_samples, num_features)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        return data, condition_mask

    def _check_data_structure(self, data, condition_mask):
        # Convert data to DataLoader

        data, condition_mask = self._check_data_shape(data, condition_mask)
        dataset_cond = TensorTupleDataset(data, condition_mask)
        num_observations = dataset_cond.__len__()

        # The scores of ALL observations must be composed together at every step,
        # so each (per-process) shard has to fit into a single batch.
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset_cond,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
            batch_size = (num_observations + self.world_size - 1) // self.world_size
            data_loader = DataLoader(
                dataset_cond,
                batch_size=batch_size,
                sampler=sampler,
                pin_memory=True,
                shuffle=False,
                drop_last=False
            )
        else:
            data_loader = DataLoader(dataset_cond, batch_size=num_observations, shuffle=False)

        return data_loader, num_observations

    def _prepare_data(self, batch, num_samples, device):
        # Expand data and condition mask to match num_samples
        data, condition_mask, idx = batch
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        data = data.unsqueeze(1).repeat(1,num_samples,1)
        condition_mask = condition_mask.unsqueeze(1).repeat(1,num_samples,1)

        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item()!=0:
            joint_data[condition_mask==1] = data.flatten()

        return joint_data, condition_mask, idx

    def _initial_sample(self, data, condition_mask):
        # Initialize latent variables with noise from the diffusion reference
        # distribution, keeping observed variables fixed. The noise on the shared
        # (hierarchy) dimensions is identical across observation rows.
        std_max = self.sde.marginal_prob_std(torch.ones_like(data))
        scale = torch.sqrt(self._hierarchy_scale(data))
        noise = self._shared_noise(data)

        data = data + std_max * scale * noise * (1 - condition_mask)
        return data

    #############################################
    # ----- Basic Sampling -----
    #############################################

    # Euler-Maruyama sampling
    def _basic_sampler(self, data, condition_mask, idx):
        """
        Basic Euler-Maruyama sampling method

        Args:
            data: Input data
                    Shape: (batch_size, num_samples, num_features)
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (batch_size, num_samples, num_features)
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.score_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.dx_t = torch.zeros(data.shape[0], self.timesteps+1, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        # Main sampling loop
        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose, total=self.timesteps-1):

            t = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            # Get score estimate
            score = self._get_score(data, t, condition_mask, idx, self.cfg_alpha)

            # Euler-Maruyama step of the reverse SDE. Integrated over one step,
            # g(t)^2 dt equals the decrease of the marginal variance sigma_m^2:
            # dx = dvar * score + sqrt(dvar) * z
            dvar = self.sde.marginal_prob_std(t)**2 - self.sde.marginal_prob_std(t_next)**2
            dx = dvar * score
            noise = self._shared_noise(data) * torch.sqrt(dvar)

            # Apply update respecting condition mask
            data = data + (dx + noise) * (1-condition_mask)

            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data
                self.dx_t[:,i] = dx
                self.score_t[:,i] = score

        return data.detach()

    # Annealed Langevin dynamics (Algorithm 1 of Geffner et al. 2023)
    def _langevin_sampler(self, data, condition_mask, idx, snr, steps_per_level):
        """
        Pure annealed Langevin dynamics: at every noise level, run `steps_per_level`
        Langevin MCMC steps targeting the bridging density. This is the sampling
        algorithm the "fnpe" composition is derived for.

        Args:
            data: Input data, shape (n_obs, num_samples, num_features)
            condition_mask: Binary mask indicating observed (1) and latent (0) values
            snr: Step size scale of the Langevin steps
            steps_per_level: Number of Langevin steps per noise level
        """
        if self.save_trajectory:
            self.data_t = torch.zeros(data.shape[0], self.timesteps, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        for i, t in tqdm.tqdm(enumerate(self.timesteps_list), disable=not self.verbose, total=self.timesteps):
            t = t.reshape(-1, 1)
            data = self._corrector_step(data, t, condition_mask, idx,
                                        steps_per_level, snr, self.cfg_alpha)
            if self.save_trajectory and i+1 < self.timesteps:
                self.data_t[:,i+1] = data

        return data.detach()

    #############################################
    # ----- Advanced Sampling -----
    #############################################

    # DPM sampling with Langevin corrector steps
    def _corrector_step(self, x, t, condition_mask, idx, steps, snr, cfg_alpha=None):
        """
        Corrector steps using Langevin dynamics

        Args:
            x: Input data
            t: Time step
            """
        # For "fnpe" the target bridging density contracts by 1/n on the shared
        # dimensions, so the Langevin step size shrinks accordingly.
        step_scale = self._hierarchy_scale(x)

        for _ in range(steps):
            # Get score estimate
            score = self._get_score(x, t, condition_mask, idx, cfg_alpha)

            # Langevin dynamics update; the noise on the shared dimensions is
            # identical across observation rows to keep them synchronized
            step_size = snr * self.sde.marginal_prob_std(t)**2 * step_scale
            noise = self._shared_noise(x) * torch.sqrt(2 * step_size)

            # Update x with the score and noise, respecting the condition mask
            grad_step = step_size * score
            x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

        return x

    def _dpm_solver_1_step(self, data_t, t, t_next, condition_mask, idx):
        """First-order solver (in noise-scale space).

        The probability-flow ODE dx = 1/2 g(t)^2 * score * dt is, in terms of the
        marginal noise scale sigma_m(t) (with g^2 = d sigma_m^2/dt), exactly
        dx = sigma_m * score * d(sigma_m) -- so the solver steps in d(sigma_m),
        not in dt.
        """
        sigma_now = self.sde.sigma_t(t)
        sigma_next = self.sde.sigma_t(t_next)
        h = sigma_now - sigma_next

        # First-order step
        score_now = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)
        data_next = data_t + h * sigma_now * score_now * (1-condition_mask)

        return data_next

    def _dpm_solver_2_step(self, data_t, t, t_next, condition_mask, idx):
        """Second-order solver (in noise-scale space)"""
        sigma_now = self.sde.sigma_t(t)
        sigma_next = self.sde.sigma_t(t_next)
        h = sigma_now - sigma_next

        # First-order step
        score_half = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)
        data_half = data_t + h * sigma_now * score_half * (1-condition_mask)

        # Second-order (Heun) step
        score_next = self._get_score(data_half, t_next, condition_mask, idx, self.cfg_alpha)
        data_next = data_t + 0.5 * h * (sigma_now * score_half + sigma_next * score_next) * (1-condition_mask)

        return data_next

    def _dpm_solver_3_step(self, data_t, t, t_next, condition_mask, idx):
        """Third-order solver (in noise-scale space)"""
        # Get sigma values at different time points
        sigma_t = self.sde.sigma_t(t)
        t_mid = (t + t_next) / 2
        sigma_mid = self.sde.sigma_t(t_mid)
        sigma_next = self.sde.sigma_t(t_next)

        # First calculate the intermediate score at time t
        score_t = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)

        # First intermediate point (Euler step)
        data_mid1 = data_t + (sigma_t - sigma_mid) * sigma_t * score_t * (1-condition_mask)

        # Get score at the first intermediate point
        score_mid1 = self._get_score(data_mid1, t_mid, condition_mask, idx, self.cfg_alpha)

        # Second intermediate point (using first intermediate)
        data_mid2 = data_t + (sigma_t - sigma_mid) * ((1/3) * sigma_t * score_t + (2/3) * sigma_mid * score_mid1) * (1-condition_mask)

        # Get score at the second intermediate point
        score_mid2 = self._get_score(data_mid2, t_mid, condition_mask, idx, self.cfg_alpha)

        # Final step using all information
        data_next = data_t + (sigma_t - sigma_next) * ((1/4) * sigma_t * score_t +
                                            (3/4) * sigma_next * score_mid2) * (1-condition_mask)

        return data_next

    def _dpm_sampler(self, data, condition_mask, idx,
                     order=2,
                     snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3):
        """
        Hybrid sampling approach combining DPM-Solver with Predictor-Corrector refinement.

        Args:
            data: Input data
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
            order: Order of DPM-Solver (1, 2 or 3)
            snr: Signal-to-noise ratio for Langevin steps
            corrector_steps_interval: Interval for applying corrector steps
            corrector_steps: Number of Langevin MCMC steps per iteration
            final_corrector_steps: Extra correction steps at the end
        """

        if self.save_trajectory:
            # Storage for trajectory (optional)
            self.data_t = torch.zeros(data.shape[0], self.timesteps, data.shape[1], data.shape[2])
            self.data_t[:,0,:,:] = data

        # Main sampling loop
        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose, total=self.timesteps-1):

            # ------- PREDICTOR: DPM-Solver -------
            t_now = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            if order == 1:
                data = self._dpm_solver_1_step(data, t_now, t_next, condition_mask, idx)
            elif order == 2:
                data = self._dpm_solver_2_step(data, t_now, t_next, condition_mask, idx)
            elif order == 3:
                data = self._dpm_solver_3_step(data, t_now, t_next, condition_mask, idx)
            else:
                raise ValueError(f"Only orders 1, 2 or 3 are supported in the DPM-Solver.")

            # ------- CORRECTOR: Langevin MCMC steps -------
            # Only apply corrector steps occasionally to save computation
            if corrector_steps > 0 and (i % corrector_steps_interval == 0 or i >= self.timesteps - final_corrector_steps):
                steps = corrector_steps
                if i >= self.timesteps - final_corrector_steps:
                    steps = corrector_steps * 5  # More steps at the end

                data = self._corrector_step(data, t_next, condition_mask, idx,
                                            steps, snr, self.cfg_alpha)

            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data

        return data.detach()
