import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import tqdm
import datetime
import os

class TensorTupleDataset(Dataset):
    def __init__(self, tensor1, tensor2, err=None):

        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.err = err
        assert len(tensor1) == len(tensor2), "Tensors must have the same length"
        
    def __len__(self):
        return len(self.tensor1)
    
    def __getitem__(self, idx):
        if self.err is not None:
            err = self.err[idx]
        else:
            err = torch.zeros_like(self.tensor1[idx])

        return self.tensor1[idx], self.tensor2[idx], idx, err

#################################################################################################
# ////////////////////////////////////////// Sampling //////////////////////////////////////////
#################################################################################################
class Sampler():
    def __init__(self, SBIm):
        self.SBIm = SBIm
        # Get SDE from model for calculations
        self.sde = self.SBIm.sde

        #############################################
    # ----- Main Sampling Loop -----
    #############################################
    
    def sample(self, world_size, data, err=None, condition_mask=None, 
               timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None,
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

        if method == "dpm":
            self.corrector_steps_interval = corrector_steps_interval
            self.corrector_steps = corrector_steps
            self.final_corrector_steps = final_corrector_steps
            self.snr = snr
            self.order = order

        if self.world_size > 1:
            manager = mp.Manager()
            result_dict = manager.dict()
            mp.spawn(self._sample_loop, args=(data, err, condition_mask, num_samples, result_dict), nprocs=self.world_size, join=True)
            samples = result_dict.get('samples', None)
            self.all_attn_weights = result_dict.get('attn_weights', None)
            manager.shutdown()

        else:
            rank = 0
            self.device = device
            samples = self._sample_loop(rank, data, err, condition_mask, num_samples)

        return samples
   
    def _sample_loop(self, rank, data, err, condition_mask, num_samples, result_dict=None):
        # Set rank
        self.rank = rank
        self.verbose = self.verbose if self.rank == 0 else False

        # Set distributed parameters
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)
        
        # Check data structure
        data_loader, self.num_observations = self._check_data_structure(data, condition_mask, err)

        # Set up timesteps on a geometric noise-scale grid (log-spaced sigma), which
        # resolves the small-noise end far better than a uniform time grid
        one = torch.ones(1, device=self.device)
        sigma_max = self.sde.marginal_prob_std(one)
        sigma_min = self.sde.marginal_prob_std(self.eps * one)
        sigmas = torch.logspace(torch.log10(sigma_max).item(), torch.log10(sigma_min).item(),
                                self.timesteps, device=self.device)
        self.timesteps_list = self.sde.time_of_sigma(sigmas)

        # Set up Attention Interpretation
        self.return_attn_weights = True
        self.attn_weights_time = self.timesteps_list[self.timesteps // 2]   # Interpretation at 50% of diffusion process
        self.all_attn_weights = []

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
                samples = self._basic_sampler(data_batch, condition_mask_batch)
            elif self.method == "dpm":
                samples = self._dpm_sampler(data_batch, condition_mask_batch,
                                            order=self.order, snr=self.snr, corrector_steps_interval=self.corrector_steps_interval,
                                            corrector_steps=self.corrector_steps, final_corrector_steps=self.final_corrector_steps)
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
            self.all_attn_weights = torch.stack(self.all_attn_weights, dim=0).to("cpu")
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
        all_attn_weights = torch.cat(self.all_attn_weights, dim=0).to(self.device)

        # Create empty tensors to gather results across processes
        gathered_samples = [torch.zeros_like(samples) for _ in range(self.world_size)]
        gathered_idx = [torch.zeros_like(indices) for _ in range(self.world_size)]
        gathered_attn_weights = [torch.zeros_like(all_attn_weights) for _ in range(self.world_size)]

        # Gather data from all processes
        dist.all_gather(gathered_samples, samples)
        dist.all_gather(gathered_idx, indices)
        dist.all_gather(gathered_attn_weights, all_attn_weights)

        if self.rank == 0:
            # Sort results by index
            gathered_idx = torch.cat(gathered_idx, dim=0)
            gathered_samples = torch.cat(gathered_samples, dim=0)
            gathered_attn_weights = torch.cat(gathered_attn_weights, dim=0)
            unique_sort_idx = [(gathered_idx == i).nonzero()[0,0].tolist() for i in gathered_idx.unique()]
            samples = gathered_samples[unique_sort_idx]

            result_dict['samples'] = samples.cpu()
            result_dict['attn_weights'] = gathered_attn_weights.cpu()

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask, cfg_alpha=None):
        """Get score estimate with optional classifier-free guidance.
        
        Handles both 2D input (num_samples, nodes_size) and 
        3D input (num_obs, num_samples, nodes_size) by flattening.
        """
        input_shape = x.shape
        
        # Flatten if 3D: (num_obs, num_samples, N) → (num_obs * num_samples, N)
        if x.dim() == 3:
            num_obs, num_samples, nodes_size = x.shape
            x_flat = x.reshape(-1, nodes_size)
            c_flat = condition_mask.reshape(-1, nodes_size)
        else:
            x_flat = x
            c_flat = condition_mask
    
        with torch.no_grad():
            # Attention weight extraction (only once per sampling run)
            if (abs(t.flatten()[0].item() - self.attn_weights_time.item()) < 1e-6 
                    and self.return_attn_weights):
                score_cond, attn_weights = self.model(
                    x=x_flat, t=t, c=c_flat, return_attn_weights=True
                )
                self.all_attn_weights.append(attn_weights)
                self.return_attn_weights = False
            else:
                score_cond = self.model(x=x_flat, t=t, c=c_flat)
    
            score_cond = self.SBIm.output_scale_function(t, score_cond)
            
            # Classifier-free guidance
            if cfg_alpha is not None:
                score_uncond = self.model(
                    x=x_flat, t=t, c=torch.zeros_like(c_flat)
                )
                score_uncond = self.SBIm.output_scale_function(t, score_uncond)
                score = score_uncond + cfg_alpha * (score_cond - score_uncond)
            else:
                score = score_cond
    
        # Unflatten back to original shape
        if len(input_shape) == 3:
            score = score.reshape(input_shape)
        
        return score

    def _check_data_shape(self, data, condition_mask, err):
        # Check data shape
        # Required shape: (num_samples, num_features)
        if len(data.shape) == 1:
            data = data.unsqueeze(0)

        # Check condition mask shape
        # Required shape: (num_samples, num_features)
        if len(condition_mask.shape) == 1:
            condition_mask = condition_mask.unsqueeze(0).repeat(data.shape[0], 1)

        # Check error shape
        # Required shape: (num_samples, num_features)
        if err is not None and len(err.shape) == 1:
            err = err.unsqueeze(0).repeat(data.shape[0], 1)

        return data, condition_mask, err

    def _check_data_structure(self, data, condition_mask, err, batch_size=1e3):
        # Convert data to DataLoader
        
        data, condition_mask, err = self._check_data_shape(data, condition_mask, err)
        dataset_cond = TensorTupleDataset(data, condition_mask, err)
        
        # Use DistributedSampler for multi-GPU
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset_cond, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False,
                drop_last=False,
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
        data, condition_mask, idx, err = batch
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        data = data.unsqueeze(1).repeat(1,num_samples,1)
        condition_mask = condition_mask.unsqueeze(1).repeat(1,num_samples,1)
        err = err.unsqueeze(1).repeat(1,num_samples,1)

        if err is not None:
            # Add scatter around the data
            err = err.to(device)
            data = torch.normal(data, err)

        joint_data = torch.zeros_like(condition_mask)
        if torch.sum(condition_mask==1).item()!=0:
            joint_data[condition_mask==1] = data.flatten()

        return joint_data, condition_mask, idx
    
    def _initial_sample(self, data, condition_mask):
        # Initialize with noise
        # Draw samples from initial noise distribution for latent variables
        # Keep observed variables fixed

        random_noise_samples = self.sde.marginal_prob_std(torch.ones_like(data)) * torch.randn_like(data) * (1-condition_mask)

        data += random_noise_samples     
        return data
  
    #############################################
    # ----- Basic Sampling -----
    #############################################
    
    # Euler-Maruyama sampling
    def _basic_sampler(self, data, condition_mask):
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

        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose):

            t = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            # Get score estimate
            score = self._get_score(data, t, condition_mask, self.cfg_alpha)

            # Euler-Maruyama step of the reverse SDE. Integrated over one step,
            # g(t)^2 dt equals the decrease of the marginal variance sigma_m^2:
            # dx = dvar * score + sqrt(dvar) * z
            dvar = self.sde.marginal_prob_std(t)**2 - self.sde.marginal_prob_std(t_next)**2
            dx = dvar * score
            noise = torch.randn_like(data) * torch.sqrt(dvar)

            # Apply update respecting condition mask
            data = data + (dx + noise) * (1-condition_mask)
            
            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1,:,:] = data
                self.dx_t[:,i,:,:] = dx
                self.score_t[:,i,:,:] = score

        return data.detach()
    
    #############################################
    # ----- Advanced Sampling -----
    #############################################
    
    # DPM sampling with Langevin corrector steps

    def _corrector_step(self, x, t, condition_mask, steps, snr, cfg_alpha=None):
        """
        Corrector steps using Langevin dynamics
        
        Args:
            x: Input data
            t: Time step
            """
        for _ in range(steps):
            # Get score estimate
            score = self._get_score(x, t, condition_mask, cfg_alpha)
            
            # Langevin dynamics update
            noise_scale = torch.sqrt(snr * 2 * self.sde.marginal_prob_std(t)**2)
            noise = torch.randn_like(x) * noise_scale
            
            # Update x with the score and noise, respecting the condition mask
            grad_step = snr * self.sde.marginal_prob_std(t)**2 * score
            x = x + grad_step * (1-condition_mask) + noise * (1-condition_mask)

        return x

    def _dpm_solver_1_step(self, data_t, t, t_next, condition_mask):
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
        score_now = self._get_score(data_t, t, condition_mask, self.cfg_alpha)
        data_next = data_t + h * sigma_now * score_now * (1-condition_mask)

        return data_next

    def _dpm_solver_2_step(self, data_t, t, t_next, condition_mask):
        """Second-order solver (in noise-scale space)"""
        sigma_now = self.sde.sigma_t(t)
        sigma_next = self.sde.sigma_t(t_next)
        h = sigma_now - sigma_next

        # First-order step
        score_half = self._get_score(data_t, t, condition_mask, self.cfg_alpha)
        data_half = data_t + h * sigma_now * score_half * (1-condition_mask)

        # Second-order (Heun) step
        score_next = self._get_score(data_half, t_next, condition_mask, self.cfg_alpha)
        data_next = data_t + 0.5 * h * (sigma_now * score_half + sigma_next * score_next) * (1-condition_mask)

        return data_next

    def _dpm_solver_3_step(self, data_t, t, t_next, condition_mask):
        """Third-order solver (in noise-scale space)"""
        # Get sigma values at different time points
        sigma_t = self.sde.sigma_t(t)
        t_mid = (t + t_next) / 2
        sigma_mid = self.sde.sigma_t(t_mid)
        sigma_next = self.sde.sigma_t(t_next)

        # First calculate the intermediate score at time t
        score_t = self._get_score(data_t, t, condition_mask, self.cfg_alpha)

        # First intermediate point (Euler step)
        data_mid1 = data_t + (sigma_t - sigma_mid) * sigma_t * score_t * (1-condition_mask)

        # Get score at the first intermediate point
        score_mid1 = self._get_score(data_mid1, t_mid, condition_mask, self.cfg_alpha)

        # Second intermediate point (using first intermediate)
        data_mid2 = data_t + (sigma_t - sigma_mid) * ((1/3) * sigma_t * score_t + (2/3) * sigma_mid * score_mid1) * (1-condition_mask)

        # Get score at the second intermediate point
        score_mid2 = self._get_score(data_mid2, t_mid, condition_mask, self.cfg_alpha)

        # Final step using all information
        data_next = data_t + (sigma_t - sigma_next) * ((1/4) * sigma_t * score_t +
                                            (3/4) * sigma_next * score_mid2) * (1-condition_mask)

        return data_next

    def _dpm_sampler(self, data, condition_mask, 
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

        for i in tqdm.tqdm(range(self.timesteps-1), disable=not self.verbose):
            
            # ------- PREDICTOR: DPM-Solver -------
            t_now = self.timesteps_list[i].reshape(-1, 1)
            t_next = self.timesteps_list[i+1].reshape(-1, 1)

            if order == 1:
                data = self._dpm_solver_1_step(data, t_now, t_next, condition_mask)
            elif order == 2:
                data = self._dpm_solver_2_step(data, t_now, t_next, condition_mask)
            elif order == 3:
                data = self._dpm_solver_3_step(data, t_now, t_next, condition_mask)
            else:
                raise ValueError(f"Only orders 1, 2 or 3 are supported in the DPM-Solver.")
            
            # ------- CORRECTOR: Langevin MCMC steps -------
            # Only apply corrector steps occasionally to save computation
            if corrector_steps > 0 and (i % corrector_steps_interval == 0 or i >= self.timesteps - final_corrector_steps):
                steps = corrector_steps
                if i >= self.timesteps - final_corrector_steps:
                    steps = corrector_steps * 2  # More steps at the end
                    
                data = self._corrector_step(data, t_next, condition_mask, 
                                            steps, snr, self.cfg_alpha)

            if self.save_trajectory:
                # Store trajectory data
                self.data_t[:,i+1] = data

        return data.detach()
    