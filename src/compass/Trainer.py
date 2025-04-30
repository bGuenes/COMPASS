import os
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import schedulefree

import tqdm
import datetime
import time

class TensorTupleDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        
    def __len__(self):
        return len(self.tensor1)
    
    def __getitem__(self, idx):
        data = self.tensor1[idx]

        if isinstance(self.tensor2, torch.distributions.Distribution):
            cond_mask = self.tensor2.sample(data.shape)
        else:
            cond_mask = self.tensor2[idx]

        return data, cond_mask, idx
    
#################################################################################################
# ////////////////////////////////////////// Training //////////////////////////////////////////
#################################################################################################
class Trainer():
    def __init__(self, SBIm):
        self.SBIm = SBIm
        #self.model_copy = copy.deepcopy(self.SBIm.model)
        # Get SDE from model for calculations
        self.sde = self.SBIm.sde

    #############################################
    # ----- Training loop -----
    #############################################
    def train(self, world_size, train_data, val_data=None,
              max_epochs=500, early_stopping_patience=20, batch_size=128, lr=1e-3,
              path=None, name="Model", device="cpu", verbose=True):
        
        """
        Training function for the score prediction task

        Args:
            rank: Rank of the current process
            world_size: Number of processes
            train_data: Training data
            val_data: Validation data
            max_epochs: Maximum number of epochs
            early_stopping_patience: Number of epochs to wait before early stopping
            batch_size: Batch size
            lr: Learning rate
            path: Path to save the model
            name: Name of the model
            device: Device to use
            verbose: Verbosity
        """
        start_time = time.time()

        # Create Checkpoint directory
        if path is None:
            path = "data/models/Model_test/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Set Parameters
        self.world_size = world_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.batch_size = batch_size
        self.lr = lr
        self.path = path
        self.name = name
        self.name_checkpoint = f"{self.name}_checkpoint"
        self.verbose = verbose
        self.eps = 1e-3 # Epsilon for numerical stability and endpoint in diffusion process

        if self.world_size > 1:
            mp.spawn(self._train_loop, args=(train_data, val_data), nprocs=self.world_size)
        else:
            rank = 0
            self.device = device
            self._train_loop(rank, train_data, val_data)

        end_time = time.time()
        training_time = (end_time-start_time) / 60
        if self.verbose:
            print(f"Training took {training_time:.1f} minutes")

    def _train_loop(self, rank, train_data, val_data):

        # Set device distribution
        if self.world_size > 1:
            self._ddp_setup(rank, self.world_size)
        else:
            self.model = self.SBIm.model.to(self.device)

        self.verbose = self.verbose if rank == 0 else False

        # Check data structure
        data_loader = self._prepare_data(train_data, batch_size=self.batch_size, rank=rank)
        if val_data is not None:
            val_loader = self._prepare_data(val_data, batch_size=1_000, rank=rank)

        # Init tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        self.train_loss = []
        self.val_loss = []

        # Set optimizer
        optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=self.lr)

        for epoch in range(self.max_epochs):
            # Train
            if self.world_size > 1: dist.barrier()

            train_loss_process = self._run_epoch(epoch, data_loader, optimizer, is_train=True)
            
            if self.world_size > 1:
                # Wait for all processes to finish training
                dist.barrier()

                # Gather losses from all processes
                epoch_train_loss = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(epoch_train_loss, torch.tensor(train_loss_process).to(self.device))
                train_loss_all = torch.mean(torch.stack(epoch_train_loss)).item()
            else:
                train_loss_all = train_loss_process

            self.train_loss.append(train_loss_all)

            # Early stopping on training loss if no validation data is provided
            if val_data is None and train_loss_all < best_val_loss:
                best_val_loss = train_loss_all
                patience_counter = 0
                if rank == 0:
                    self._save_checkpoint(name=self.name_checkpoint)
            elif val_data is None and train_loss_all >= best_val_loss:
                patience_counter += 1


            # Validate
            if val_data is not None:
                val_loss_process = self._run_epoch(epoch, val_loader, optimizer, is_train=False)

                if self.world_size > 1:
                    # Wait for all processes to finish validation
                    dist.barrier()

                    # Gather losses from all processes
                    epoch_val_loss = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
                    dist.all_gather(epoch_val_loss, torch.tensor(val_loss_process).to(self.device))
                    val_loss_all = torch.mean(torch.stack(epoch_val_loss)).item()
                else:
                    val_loss_all = val_loss_process

                self.val_loss.append(val_loss_all)

                # Early stopping
                if val_loss_all < best_val_loss:
                    best_val_loss = val_loss_all
                    patience_counter = 0
                    if rank == 0:
                        self._save_checkpoint(name=self.name_checkpoint)
                elif val_loss_all >= best_val_loss:
                    patience_counter += 1

            # Wait for all processes to finish validation
            if self.world_size > 1: dist.barrier()

            if self.verbose:
                if val_data is not None:
                    print(f'--- Epoch: {epoch+1:3d} --- Training Loss: {train_loss_all:8.3f} --- Validation Loss: {val_loss_all:8.3f} ---')
                else:
                    print(f'--- Epoch: {epoch+1:3d} --- Training Loss: {train_loss_all:8.3f} ---')
                print()
                time.sleep(0.2)

            if patience_counter == self.early_stopping_patience:
                    break
            
            
        if self.world_size > 1:
            dist.barrier()
            if rank == 0:
                self._save_checkpoint(name=self.name)
            dist.destroy_process_group()
        
    def _run_epoch(self, epoch, data_loader, optimizer, is_train):
        if self.world_size > 1:
            data_loader.sampler.set_epoch(epoch)

        if is_train:
            self.model.train()
            optimizer.train()
        else:
            self.model.eval()
            optimizer.eval()

        total_loss = 0
        batch_count = 0

        show_progress = self.verbose if is_train else False
        for batch in tqdm.tqdm(data_loader, disable=not show_progress):
            if is_train:
                optimizer.zero_grad()
            
            loss = self._run_batch(batch)
            total_loss += loss.item() / batch[0].shape[0]
            batch_count += 1

            if is_train:
                loss.backward()
                if self.world_size > 1: dist.barrier()
                optimizer.step()

        return total_loss / batch_count

    def _run_batch(self, batch):
        # Get data
        data, condition_mask, idx = self._prepare_batch(batch, self.device)

        # Get timesteps
        timesteps = torch.rand(data.shape[0], 1, device=self.device) * (1.-self.eps)+self.eps

        # Sample x_1 from noise distribution
        x_1 = torch.randn_like(data)*(1-condition_mask) + data*condition_mask
        # Calculate x at time t in diffusion process
        x_t = self.SBIm.forward_diffusion_sample(data, timesteps, x_1, condition_mask)
        # Get score
        score = self._get_score(x_t, timesteps, condition_mask)
        # Calculate loss
        loss = self.loss_fn(score, timesteps, x_1, condition_mask)

        return loss

    #############################################
    # ----- Loss Function -----
    #############################################
    def loss_fn(self, score, timestep, x_1, condition_mask):
        '''
        Loss function for the score prediction task

        Args:
            score: Predicted score
                    
        The target is the noise added to the data at a specific timestep 
        Meaning the prediction is the approximation of the noise added to the data
        '''
        sigma_t = self.sde.marginal_prob_std(timestep).unsqueeze(1).to(score.device)
        x_1 = x_1.unsqueeze(2).to(score.device)
        condition_mask = condition_mask.unsqueeze(2).to(score.device)
        score = score.unsqueeze(2)

        loss = torch.mean(sigma_t**2 * torch.sum((1-condition_mask)*(x_1+sigma_t*score)**2))

        return loss
    
    #############################################
    # ----- Multi-GPU setup -----
    #############################################
    def _ddp_setup(self, rank, world_size):

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

        self.device = torch.device(f'cuda:{rank}')
        self.SBIm.model.to(self.device)
        self.model = DDP(self.SBIm.model, device_ids=[rank], output_device=rank)

    #############################################
    # ----- Standard Functions -----
    #############################################

    def _get_score(self, x, t, condition_mask):
        """Get score estimate from model"""
        # Get conditional score
        out = self.model(x=x, t=t, c=condition_mask)
        score = self.SBIm.output_scale_function(t, out)
                
        return score

    def _prepare_batch(self, batch, device):
        # Expand data and condition mask to match num_samples
        data, condition_mask, idx = batch
        data = data.to(device)
        condition_mask = condition_mask.to(device)

        return data, condition_mask, idx

    def _prepare_data(self, data, batch_size, rank):
        condition_mask = torch.distributions.bernoulli.Bernoulli(0.33)

        dataset = TensorTupleDataset(data, condition_mask)

        if self.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank, shuffle=True)
            data_loader = DataLoader(
                dataset, 
                batch_size=batch_size,
                pin_memory=True,
                shuffle=False,
                sampler=sampler
            )
        else:
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader

    def _save_checkpoint(self, name="Model_checkpoint"):
        # Save checkpoint
        if self.world_size > 1:
            # Save torch model in case of multi-GPU training
            state_dict = {
                'model_state_dict' : self.model.module.state_dict(),
                'nodes_size': self.SBIm.nodes_size,
                'sde_type': self.SBIm.sde_type,
                'sigma': self.SBIm.sigma,
                'hidden_size': self.SBIm.hidden_size,
                'depth': self.SBIm.depth,
                'num_heads': self.SBIm.num_heads,
                'mlp_ratio': self.SBIm.mlp_ratio
            }
            torch.save(state_dict, f"{self.path}/{name}.pt")
        else:
            self.SBIm.save(path=self.path ,name=name)
