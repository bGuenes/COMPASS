from compass import ScoreBasedInferenceModel as SBIm

import torch

from scipy.stats import norm
from scipy.stats import gaussian_kde

import numpy as np
import tarp

import optuna
from optuna.study import MaxTrialsCallback

import argparse
import os

# ---------------------------------------------------------------
# Data generation
def gen_data(int):
    theta1 = 3 * torch.randn(int)
    x1 = 2 * torch.sin(theta1) + torch.randn(int) * 0.5
    x2 = 0.1 * theta1**2 + 0.5*torch.abs(x1) * torch.randn(int)

    return theta1.unsqueeze(1), torch.stack([x1, x2],dim=1)

# ---------------------------------------------------------------
# Optuna objective function
def objective(trial):
    try:
        # Variables
        batch_size = trial.suggest_int('batch_size', 16,1024)
        sigma = trial.suggest_float('sigma', 1.1, 30.0)
        depth = trial.suggest_int('depth', 1, 12)
        num_heads = trial.suggest_int('num_heads', 1, 32)
        hidden_size_factor = trial.suggest_int('hidden_size_factor', 1,256)
        hidden_size = num_heads*hidden_size_factor
        mlp_ratio = trial.suggest_int('mlp_ratio', 1, 10)

        # Load data
        train_theta, train_x = gen_data(100_000)
        val_theta, val_x = gen_data(10_000)

        print(f"Batch size: {batch_size}, Sigma: {sigma}, Depth: {depth}, Num heads: {num_heads}, Hidden size factor: {hidden_size_factor}, MLP ratio: {mlp_ratio}")

        # Setup model
        nodes_size = 3
        sbim = SBIm(nodes_size=nodes_size, sigma=sigma, depth=depth, hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # Train model
        sbim.train(train_theta, train_x, val_theta, val_x, batch_size=batch_size, max_epochs=5, device="cuda", verbose=True, path="data/tutorial_Gaussians", early_stopping_patience=20)

        # Evaluate model
        test_theta, test_x = gen_data(100)

        hat_theta = sbim.sample(test_x, verbose=True, num_samples=1000, timesteps=100, device="cuda")
        
        hat_theta = hat_theta.contiguous().cpu().numpy()
        test_theta = test_theta.cpu().numpy()

        # Log Prob
        def calc_log_prob(samples, theta):
            try:
                kde = gaussian_kde(samples.T)
                return kde.logpdf(theta).item()
            except:
                return -1e20
            
        log_probs = np.array([calc_log_prob(hat_theta[i], test_theta[i]) for i in range(len(hat_theta))])
        mean_log_prob = -np.mean(log_probs)

        # measure tarp
        ecp, alpha = tarp.get_tarp_coverage(
            hat_theta.transpose(1,0,2), test_theta,
            norm=True, bootstrap=True,
            num_bootstrap=100
        )
        tarp_diff = np.abs(ecp-np.linspace(0,1,ecp.shape[1])).max()

        print(f"Mean log prob: {mean_log_prob}, TARP diff: {tarp_diff.item()}")
        return mean_log_prob, tarp_diff.item()

    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return very bad scores for failed trials
        return float('inf'), float('inf')


# ---------------------------------------------------------------
# Main function
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()

    available_gpus = ','.join(map(str, range(torch.cuda.device_count())))
    parser.add_argument('--gpus', type=str, default=available_gpus, help=f'comma-separated list of GPU ids to use (default: {available_gpus})')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    world_size = len(args.gpus.split(','))

    # Optuna
    study_name = 'tutorial_Gaussians'  # Unique identifier of the study.
    storage_name = 'sqlite:///tutorial_Gaussians.db'
    study = optuna.create_study(study_name=study_name, storage=storage_name,directions=['minimize', 'minimize'], load_if_exists=True)
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    study.optimize(objective, callbacks=[MaxTrialsCallback(200)])
