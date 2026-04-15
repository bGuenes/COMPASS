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
    def __init__(self, SBIm):
        self.SBIm = SBIm
        # Get SDE from model for calculations
        self.sde = self.SBIm.sde

    #############################################
    # ----- Main Sampling Loop -----
    #############################################
    
    def sample(self, world_size, data, condition_mask=None, timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, hierarchy=None,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cpu", verbose=True, method="dpm", save_trajectory=False, result_dict=None):
        """
        Sample from the model using the specified method

        Args:
            data: Input data
                    - Should be a DataLoader or a tuple of (data, condition_mask)
                        - Shape data: (num_samples, num_observed_features)
                        - Shape condition_mask: (num_samples, num_total_features)
                    - Can also be a single tensor of data, in which case condition_mask must be provided
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_samples, num_total_features)
            timesteps: Number of diffusion steps
            eps: End time for diffusion process
            num_samples: Number of samples to generate
            cfg_alpha: Classifier-free guidance strength
            hierarchy: Hierarchy of variables for multi-observation sampling
                    - List of indices for which variables compositional score modeling should be applied to

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

        if method == "dpm":
            self.corrector_steps_interval = corrector_steps_interval
            self.corrector_steps = corrector_steps
            self.final_corrector_steps = final_corrector_steps
            self.snr = snr
            self.order = order

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

        if condition_mask is None:
            raise ValueError("condition_mask must be provided for multi-observation compositional score modeling.")

        # Set distributed parameters
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)

        # Set hierarchy for compositional score modeling to all latent if not provided
        if self.hierarchy is None:
            base_mask = condition_mask[0] if condition_mask.dim() > 1 else condition_mask
            self.hierarchy = torch.where(base_mask == 0)[0].tolist()
        
        # Check data structure
        data_loader, self.num_observations = self._check_data_structure(data, condition_mask)

        # Set up timesteps
        self.timesteps_list = torch.linspace(1., self.eps, self.timesteps, device=self.device)
        self.dt = self.timesteps_list[0] - self.timesteps_list[1]
        
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
            elif self.method == "multi_observation":
                samples = self._multi_observation_sampler(data_batch, condition_mask_batch)
            else:
                raise ValueError(f"Sampling method {self.method} not recognized.")

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
            _, sort_idx = gathered_idx.sort()
            samples = gathered_samples[sort_idx]

            result_dict['samples'] = samples.cpu()

    def _gather_scores(self, score_table, indices):
        # Gather scores from all processes
        gathered_scores = [torch.zeros_like(score_table) for _ in range(self.world_size)]
        
        indices = indices.to(self.device)
        gathered_idx = [torch.zeros_like(indices) for _ in range(self.world_size)]

        dist.all_gather(gathered_scores, score_table)
        dist.all_gather(gathered_idx, indices)

        # Sort results by index
        gathered_idx = torch.cat(gathered_idx, dim=0)
        gathered_scores = torch.cat(gathered_scores, dim=0)
        _, sort_idx = gathered_idx.sort()
        gathered_scores = gathered_scores[sort_idx]

        return gathered_scores

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask, indices, cfg_alpha=None):
        """Get score estimate with optional classifier-free guidance"""
        with torch.no_grad():
            n_obs, num_samples, num_features = x.shape

            # Flatten (n_obs, num_samples, num_features) → (n_obs*num_samples, num_features)
            # so all observations are processed in a single forward pass
            x_flat = x.reshape(n_obs * num_samples, num_features)
            c_flat = condition_mask.reshape(n_obs * num_samples, num_features)

            if cfg_alpha is not None:
                # Run conditional + unconditional in a single double-batch forward pass
                x_double = torch.cat([x_flat, x_flat], dim=0)
                c_double = torch.cat([c_flat, torch.zeros_like(c_flat)], dim=0)
                scores_double = self.SBIm.model(x=x_double, t=t, c=c_double)
                scores_double = self.SBIm.output_scale_function(t, scores_double)
                scores_cond, scores_uncond = scores_double.chunk(2, dim=0)
                scores_flat = scores_uncond + cfg_alpha * (scores_cond - scores_uncond)
            else:
                scores_flat = self.SBIm.model(x=x_flat, t=t, c=c_flat)
                scores_flat = self.SBIm.output_scale_function(t, scores_flat)

            score_table = scores_flat.reshape(n_obs, num_samples, num_features)

            if self.world_size > 1:
                dist.barrier()
                score_table = self._gather_scores(score_table, indices)
                x_for_composition = self._gather_scores(x, indices)
            else:
                x_for_composition = x

            score = self._compositional_score(score_table, x_for_composition, t, self.hierarchy)

        if self.world_size > 1:
            return score[indices]
        return score
    
    def _compositional_score(self, scores, x, t, hierarchy):
        """
        F-NPSE compositional score (Geffner et al. 2023, Eq. 7):
            s_composed(θ, t) = (1-n)(T-t)/T * ∇log p(θ) + Σ_j s_ψ(θ, t, x_j)
        Only applied on the `hierarchy` dimensions — i.e. the parameters shared
        across all n observations. Other latent dimensions keep their per-obs
        score since they are not shared.

        With t ∈ [eps, 1] and T normalised so (T-t)/T = 1-t.
        """
        n_obs = scores.shape[0]

        # Prior on hierarchy dims (Gaussian). Allocate only on those dims.
        mu_h = x.new_tensor([-2.3, -2.89])
        sigma_h = x.new_tensor([0.3, 0.3])

        # We use the *diffused* prior variance σ² + std(t)² as a numerical
        # stabiliser — under VESDE with σ=25, std(t) dominates σ_h at high t
        # and the undiffused score -(x-µ)/σ² explodes. This corresponds to
        # the equivalent construction p_t^f ∝ p_t(θ)^((1-n)(T-t)/T) Π p_t(θ|x_j),
        # where the prior is diffused alongside the observation posteriors.
        marginal_std = self.sde.marginal_prob_std(t)
        diffused_var_h = sigma_h**2 + marginal_std**2

        # x is synchronised across observations on hierarchy dims, so a single
        # copy suffices. Shape: (1, num_samples, |hierarchy|).
        x_h = x[:1, :, hierarchy]
        prior_score_h = -(x_h - mu_h) / diffused_var_h

        # SUM (not mean!) of per-observation scores on hierarchy dims.
        sum_scores_h = torch.sum(scores[:, :, hierarchy], dim=0, keepdim=True)

        # Prefactor (1-n)(1-t) — NOT (1/n - 1)(1-t), which would give the
        # correct composed score divided by n and bias all samplers.
        prefactor = (1.0 - n_obs) * (1.0 - t)

        # Broadcast across the n_obs axis on assignment — all observations
        # receive the same compositional score on the shared parameters.
        scores[:, :, hierarchy] = prefactor * prior_score_h + sum_scores_h

        return scores

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

    def _check_data_structure(self, data, condition_mask, batch_size=1e3):
        # Convert data to DataLoader
        
        data, condition_mask = self._check_data_shape(data, condition_mask)
        dataset_cond = TensorTupleDataset(data, condition_mask)
        
        # Use DistributedSampler for multi-GPU
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset_cond, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False,
                drop_last=False
            )
            data_loader = DataLoader(
                dataset_cond, 
                batch_size=int(batch_size), 
                sampler=sampler, 
                pin_memory=True,
                shuffle=False,
                drop_last=False
            )
        else:
            data_loader = DataLoader(dataset_cond, batch_size=int(batch_size), shuffle=False)
        
        num_observations = dataset_cond.__len__()

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
        # Initialize with noise
        # Draw samples from initial noise distribution for latent variables
        # Keep observed variables fixed

        # In case for compositional score modeling the random samples need to be the same across all samples
        if self.rank == 0:
            self.prior_score = torch.randn(data.shape[1], data.shape[2], device=self.device).unsqueeze(0).repeat(data.shape[0],1,1) * (1-condition_mask)
        else:
            # If not rank 0 get self.prior_score from rank 0
            self.prior_score = torch.zeros(data.shape[0], data.shape[1], data.shape[2], device=self.device)

        if self.world_size > 1:
            dist.barrier()
            dist.broadcast(self.prior_score, src=0)

        random_noise_samples = self.sde.marginal_prob_std(torch.ones_like(data)) * self.prior_score

        data += random_noise_samples
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
        for i, t in tqdm.tqdm(enumerate(self.timesteps_list), disable=not self.verbose, total=self.timesteps):

            t = t.reshape(-1, 1)
            
            # Get score estimate
            score = self._get_score(data, t, condition_mask, idx, self.cfg_alpha)
            
            # Update step
            dx = self.sde.sigma**(2*t) * score * self.dt
            
            # Apply update respecting condition mask
            data = data + dx * (1-condition_mask)
            
            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data
                self.dx_t[:,i] = dx
                self.score_t[:,i] = score

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
        for _ in range(steps):
            # Get score estimate
            score = self._get_score(x, t, condition_mask, idx, cfg_alpha)
            
            # Langevin dynamics update
            noise_scale = torch.sqrt(snr * 2 * self.sde.marginal_prob_std(t)**2)
            noise = torch.randn_like(x) * noise_scale
            
            # Update x with the score and noise, respecting the condition mask
            grad_step = snr * self.sde.marginal_prob_std(t)**2 * score
            x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

        return x

    def _dpm_solver_1_step(self, data_t, t, t_next, condition_mask, idx):
        """First-order solver"""
        sigma_now = self.sde.sigma_t(t)

        # First-order step
        score_now = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)
        data_next = data_t + (t-t_next) * sigma_now * score_now * (1-condition_mask)

        return data_next
    
    def _dpm_solver_2_step(self, data_t, t, t_next, condition_mask, idx):
        """Second-order solver"""
        sigma_now = self.sde.sigma_t(t)
        sigma_next = self.sde.sigma_t(t_next)

        # First-order step
        score_half = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)
        data_half = data_t + (t-t_next) * sigma_now * score_half * (1-condition_mask)

        # Second-order step
        score_next = self._get_score(data_half, t_next, condition_mask, idx, self.cfg_alpha)
        data_next = data_t + 0.5 * (t-t_next) * (sigma_now**2 * score_half + sigma_next**2 * score_next) * (1-condition_mask)

        return data_next
    
    def _dpm_solver_3_step(self, data_t, t, t_next, condition_mask, idx):
        """Third-order solver"""
        # Get sigma values at different time points
        sigma_t = self.sde.sigma_t(t)
        t_mid = (t + t_next) / 2
        sigma_mid = self.sde.sigma_t(t_mid)

        # First calculate the intermediate score at time t
        score_t = self._get_score(data_t, t, condition_mask, idx, self.cfg_alpha)

        # First intermediate point (Euler step)
        data_mid1 = data_t + (t - t_mid) * sigma_t * score_t * (1-condition_mask)

        # Get score at the first intermediate point
        score_mid1 = self._get_score(data_mid1, t_mid, condition_mask, idx, self.cfg_alpha)

        # Second intermediate point (using first intermediate)
        data_mid2 = data_t + (t - t_mid) * ((1/3) * sigma_t * score_t + (2/3) * sigma_mid * score_mid1) * (1-condition_mask)

        # Get score at the second intermediate point
        score_mid2 = self._get_score(data_mid2, t_mid, condition_mask, idx, self.cfg_alpha)

        # Final step: score_mid2 is evaluated at t_mid, so weight with sigma_mid (not sigma_next)
        data_next = data_t + (t - t_next) * ((1/4) * sigma_t * score_t +
                                            (3/4) * sigma_mid * score_mid2) * (1-condition_mask)

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
    