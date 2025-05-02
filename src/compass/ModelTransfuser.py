import os
import sys
import pickle

import torch
import torch.nn as nn

import numpy as np

import scipy
from scipy import optimize
from scipy.stats import norm, gaussian_kde

import matplotlib.pyplot as plt
import seaborn as sns

from .ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm

#################################################################################################
# ///////////////////////////////////// Model Comparison ////////////////////////////////////////
#################################################################################################

class ModelTransfuser():
    def __init__(self, path=None):
        
        ## Check if the path exists
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
        self.path = path
        
        self.models_dict = {}
        self.data_dict = {}
        self.trained_models = False # Flag to check if models are trained

    #############################################
    # ----- Model Management -----
    #############################################

    # Add a trained model to the transfuser
    def add_model(self, model_name, model):
        """
        Add a trained model to the transfuser.

        Args:
            model_name: The name of the model.
            model: The model itself.
        """
        self.models_dict[model_name] = model
        self.trained_models = True
        print(f"Model {model_name} added to transfuser.")

    # Add multiple trained models to the transfuser
    def add_models(self, models_dict):
        """
        Add multiple trained models to the transfuser.

        Args:
            models_dict: A dictionary of models to add.
        """
        for model_name, model in models_dict.items():
            self.add_model(model_name, model)

        self.trained_models = True
        print("All models added to transfuser.")

    # Add data to a model
    def add_data(self, model_name, theta, x, val_theta=None, val_x=None):
        """
        Add training and validation data to a model.

        Args:
            model_name: The name of the model.
            train_data: The training data.
            val_data: The validation data (optional).
        """
        if val_theta is None:
            self.data_dict[model_name] = {
                "train_theta": theta,
                "train_x": x,
            }
        else:
            self.data_dict[model_name] = {
                "train_theta": theta,
                "train_x": x,
                "val_theta": val_theta,
                "val_x": val_x,
            }
        self.trained_models = False
        print(f"Data added to model {model_name}")

    # Remove a model from the transfuser
    def remove_model(self, model_name):
        """
        Remove a model from the transfuser.

        Args:
            model_name: The name of the model to remove.
        """
        if model_name in self.models_dict:
            del self.models_dict[model_name]
            print(f"Model {model_name} removed from transfuser.")
        else:
            print(f"Model {model_name} not found in transfuser.")

    #############################################
    # ----- Initialize Models -----
    #############################################

    def init_models(self, sde_type, sigma, hidden_size, depth, num_heads, mlp_ratio):
        """
        Initialize the Score-Based Inference Models with the given parameters

        Args:
            sde_type: The type of SDE
            sigma: The sigma value
            hidden_size: The size of the hidden layer
            depth: The depth of the model
            num_heads: The number of heads in the model
            mlp_ratio: The MLP ratio
        """

        if self.trained_models:
            print("Models are already trained. This will overwrite the models.")
            return
        else:
            init_models = []
            for model_name in self.data_dict.keys():
                nodes_size = self.data_dict[model_name]["train_theta"].shape[1] + self.data_dict[model_name]["train_x"].shape[1]
                self.models_dict[model_name] = SBIm(nodes_size=nodes_size,
                                                    sde_type=sde_type,
                                                    sigma=sigma,
                                                    hidden_size=hidden_size,
                                                    depth=depth,
                                                    num_heads=num_heads,
                                                    mlp_ratio=mlp_ratio)
                init_models.append(model_name)

            print(f"Models initialized: {init_models}")

    #############################################
    # ----- Train Models -----
    #############################################

    def train_models(self, batch_size=128, max_epochs=500, lr=1e-3, device="cuda",
                verbose=False, path=None, early_stopping_patience=20): 
        
        """
        Train the models on the provided data

        Args:
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            lr: Learning rate
            device: Device to run training on
                    if "cuda", training will be distributed across all available GPUs
            verbose: Whether to show training progress
            path: Path to save model
            early_stopping_patience: Number of epochs to wait before early stopping
        """

        if self.trained_models:
            print("Continue training existing models.")

        if path is not None:
            self.path = path
        elif self.path is not None:
            path = self.path

        for model_name in self.models_dict.keys():
            model = self.models_dict[model_name]

            theta = self.data_dict[model_name]["train_theta"]
            x = self.data_dict[model_name]["train_x"]
            val_theta = self.data_dict[model_name].get("val_theta", None)
            val_x = self.data_dict[model_name].get("val_x", None)

            model.train(theta=theta, x=x, theta_val=val_theta, x_val=val_x,
                        batch_size=batch_size, max_epochs=max_epochs, lr=lr, device=device,
                        verbose=verbose, path=path, name=model_name ,early_stopping_patience=early_stopping_patience)

            load_path = f"{path}/{model_name}.pt"
            self.models_dict[model_name] = SBIm.load(path=load_path, device="cpu")
            print(f"Model {model_name} trained")
            torch.cuda.empty_cache()

        self.trained_models = True

    #############################################
    # ----- Model Comparison -----
    #############################################

    def compare(self, x, err=None, condition_mask=None,
               timesteps=50, eps=1e-3, num_samples=1000, cfg_alpha=None, multi_obs_inference=False, hierarchy=None,
               order=2, snr=0.1, corrector_steps_interval=5, corrector_steps=5, final_corrector_steps=3,
               device="cuda", verbose=False, method="dpm"):
        """
        Compare the models on the provided observations

        Args:
            observations: The observations to compare the models on
            condition_mask: Binary mask indicating observed values (1) and latent values (0)
                    Shape: (num_samples, num_total_features)
                    Optional
            timesteps: Number of timesteps for the model
            eps: Epsilon value for the model
            num_samples: Number of samples to generate
            cfg_alpha: CFG alpha value for the model
            multi_obs_inference: Whether to use multi-observation inference
            hierarchy: Hierarchy for the model
            order: Order of the model
            snr: Signal-to-noise ratio for the model
            corrector_steps_interval: Corrector steps interval for the model
            corrector_steps: Corrector steps for the model
            final_corrector_steps: Final corrector steps for the model
            device: Device to run inference on
            verbose: Whether to show inference progress
            method: Method to use for inference
        """

        if not self.trained_models:
            print("Models are not trained or provided. Please train the models before comparing.")
            return
        
        self.stats = {}
        self.model_null_log_probs = {}
        self.softmax = nn.Softmax(dim=0)
        
        for model_name, model in self.models_dict.items():
            self.stats[model_name] = {}
            if condition_mask is None:
                condition_mask = torch.cat([torch.zeros(model.nodes_size-x.shape[-1]),torch.ones(x.shape[-1])])
            ####################
            # Posterior sampling
            posterior_samples = model.sample(x=x, err=err, condition_mask=condition_mask,
                                            timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method)
            posterior_samples = posterior_samples.cpu().numpy()

            # MAP estimation
            theta_hat = np.array([self._map_kde(posterior_samples[i]) for i in range(len(posterior_samples))])
            MAP_posterior, std_MAP_posterior = torch.tensor(theta_hat[:,0], dtype=torch.float), torch.tensor(theta_hat[:,1], dtype=torch.float)

            # Storing MAP and std MAP
            self.stats[model_name]["MAP"] = theta_hat

            ####################
            # Null Hypothesis
            null_samples = model.sample(timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method)
            null_samples = null_samples[0,:,condition_mask.bool()].cpu().numpy()

            # Log probability of null hypothesis
            null_log_probs = torch.tensor([self._log_prob(null_samples, obs) for obs in x])
            self.stats[model_name]["log_probs_nullHyp"] = null_log_probs

            ####################
            # Likelihood sampling
            likelihood_samples = model.sample(theta=MAP_posterior, err=std_MAP_posterior, condition_mask=(1-condition_mask),
                                            timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method)
            likelihood_samples = likelihood_samples.cpu().numpy()

            # Log probability of likelihood
            log_probs = torch.tensor([self._log_prob(likelihood_samples[i], x[i]) for i in range(len(x))])
            self.stats[model_name]["log_probs"] = log_probs
            self.stats[model_name]["AIC"] = log_probs.sum() 

            # Null Hypothesis test
            self.stats[model_name]["Bayes_Factor_Null_Hyp"] = log_probs.sum() - null_log_probs.sum()

        # Calculate Model Probabilitys from AICs
        aics = [self.stats[model_name]["AIC"] for model_name in self.stats.keys()]
        aics = torch.tensor(aics)
        model_probs = self.softmax(aics)

        # Calculate Probability of each observation
        log_probs = torch.stack([self.stats[model_name]["log_probs"] for model_name in self.stats.keys()])
        probs = self.softmax(log_probs)

        for i, model_name in enumerate(self.stats.keys()):
            self.stats[model_name]["model_prob"] = model_probs[i].item()
            self.stats[model_name]["obs_probs"] = probs[i]

        model_names = list(self.stats.keys())
        best_model = model_names[model_probs.argmax()]
        best_model_prob = 100*model_probs.max()

        # Null Hypothesis test
        best_bayes_factor = self.stats[best_model]["Bayes_Factor_Null_Hyp"]
        hypothesis_test = "could" if best_bayes_factor > 0 else "could not"
        hypothesis_test_strength = self._bayes_factor_strength(best_bayes_factor)

        model_print_length = len(max(model_names, key=len))
        print(f"Probabilities of the models after {len(x)} observations:")
        for model in model_names:
            print(f"{model.ljust(model_print_length)}: {100*self.stats[model]['model_prob']:6.2f} %")
        print()
        print(f"Model {best_model} fits the data best " + 
                f"with a relative support of {best_model_prob:.1f}% among the considered models "+
                f"and {hypothesis_test} reject the null hypothesis{hypothesis_test_strength}.")
        
        if self.path is not None:
            with open(f"{self.path}/model_comp.pkl", "wb") as f:
                pickle.dump(self.stats, f)

    #############################################
    # ----- Kernel Density Estimation -----
    #############################################

    def _log_prob(self, samples, observation):
        """Compute the log probability of the samples"""
        kde = gaussian_kde(samples.T)
        log_prob = kde.logpdf(observation).item()
        return log_prob

    def _map_kde(self, samples):
        """Find the joint mode of the multivariate distribution"""
        kde = gaussian_kde(samples.T)  # KDE expects (n_dims, n_samples)
        
        # Start optimization from the mean
        initial_guess = np.mean(samples, axis=0)
        
        # Use full minimize with multiple dimensions
        result = optimize.minimize(lambda x: -kde(x.reshape(-1, 1)), initial_guess)
        std_devs = np.sqrt(np.diag(kde.covariance))

        return result.x, std_devs
    
    ##############################################
    # ----- Bayes Factor Strenght -----
    ##############################################

    def _bayes_factor_strength(self, BF):
        """
        Calculate the strength of the Bayes factor.

        Args:
            BF: The Log Bayes factor.

        Returns:
            The strength of the Bayes factor as a string.
        """
        hypothesis_test_strength = torch.exp(BF)

        if 1 < hypothesis_test_strength <= 3.2:
            hypothesis_test_strength = " barley"
        elif 3.2 < hypothesis_test_strength <= 10:
            hypothesis_test_strength = " substantially"
        elif 10 < hypothesis_test_strength <= 100:
            hypothesis_test_strength = " strongly"
        elif 100 < hypothesis_test_strength:
            hypothesis_test_strength = " decisively"
        else:
            hypothesis_test_strength = ""

        return hypothesis_test_strength
    
    ##############################################
    # ----- Plotting -----
    ##############################################

    def plots(self, stats_dict=None, n_models=10, sort="median", path=None, show=True):

        # Check path
        if path is None:
            path = self.path
            if not os.path.exists(path):
                os.makedirs(path)

        # Check stats_dict
        if stats_dict is None:
            stats_dict = self.stats

        # Sort models by log_probs
        if sort == "median":
            sorted_models = sorted(stats_dict, key=lambda x: stats_dict[x]["log_probs"].median(),reverse=True)
        elif sort == "mean":
            sorted_models = sorted(stats_dict, key=lambda x: stats_dict[x]["log_probs"].mean(),reverse=True)
        elif type(sort) == list:
            sorted_models = sort
            # add the remaining models to the end of the list for correct probability calculation
            for model in stats_dict.keys():
                if model not in sorted_models:
                    sorted_models.append(model)
        elif sort == "none":
            sorted_models = list(stats_dict.keys())
        stats_dict = {model: stats_dict[model] for model in sorted_models}

        model_names = list(stats_dict.keys())
        model_probs = torch.tensor([stats_dict[model]["model_prob"] for model in model_names])
        model_log_probs = torch.stack([stats_dict[model]["log_probs"] for model in model_names])
        model_obs_probs = torch.stack([stats_dict[model]["obs_probs"] for model in model_names])

        sns.set_context("paper")

        # Plot violin plot of model probabilities
        plt.figure(figsize=(10, 5))
        sns.violinplot(data=model_obs_probs.T[:,:n_models])
        plt.xticks(ticks=range(n_models), labels=model_names[:n_models], rotation=45, ha='right')
        plt.title("Model Probabilities")
        plt.xlabel("Model")
        plt.ylabel("Probability of Observations")
        if path is not None:
            plt.savefig(f"{path}/model_probs_violin.png")
        if show:
            plt.show()
        plt.close()

        # Plot updated model probabilities
        plt.figure(figsize=(10, 5))
        plt.plot(torch.arange(1, model_log_probs.shape[1]+1).repeat(n_models,1).T,
                    torch.nn.functional.softmax(model_log_probs.cumsum(1),0).T[:,:n_models],
                    label=model_names[:n_models], marker='o', markersize=3, linewidth=1)
        plt.legend()
        #plt.xscale("log")
        plt.title("Updated Model Probabilities")
        plt.xlabel("Number of observations")
        plt.ylabel("Model Probability")
        plt.grid(True)
        if path is not None:
            plt.savefig(f"{path}/model_probs_updated.png")
        if show:
            plt.show()
        plt.close()
