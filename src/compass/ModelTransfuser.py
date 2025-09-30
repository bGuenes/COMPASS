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
from matplotlib.colors import PowerNorm

import seaborn as sns

from .ScoreBasedInferenceModel import ScoreBasedInferenceModel as SBIm

from itertools import compress
import warnings
warnings.filterwarnings('ignore')

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

    #---------------------------
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

    #---------------------------
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

    #---------------------------
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

    #---------------------------
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
        Compare the models on the provided observations.
        The results are saved in the self.stats dictionary and the provided path.

        Args:
            x:              The observations to compare the models on.
                                Shape: (num_samples, num_obs_features)
            err:            (optional) The observation uncertainties. If not provided, it is assumed to be zero.
                                Shape: (num_samples, num_obs_features)
            condition_mask: (optional) Binary mask indicating observed values (1) and latent values (0).
                                Should be provided if the there are missing observations in the data.
                                If not provided, it is assumed, that 
                                Shape: (num_samples, num_total_features)
            timesteps:      Number of timesteps for the diffusion process.
                                (default) - 50 timesteps
            eps:            Epsilon value for the model
            num_samples:    Number of samples to generate
            cfg_alpha:      CFG alpha value for the model
            multi_obs_inference: Whether to use multi-observation inference
            hierarchy:      Hierarchy for the model
            order:          Order of the model
            snr:            Signal-to-noise ratio for the model
            corrector_steps_interval: Corrector steps interval for the model
            corrector_steps: Corrector steps for the model
            final_corrector_steps: Final corrector steps for the model
            device:         Device to run inference on
            verbose:        (bool) Whether to show inference progress
            method:         (string) Method used to solve the SDE during inference.
                                "dpm"   - (default) Using the DPM-Solver for infernce with order 'order'
                                "euler" - Using the Euler-Maruyama method for inference
        """

        if not self.trained_models:
            print("Models are not trained or provided. Please train the models before comparing.")
            return
        
        self.stats = {}
        self.model_null_log_probs = {}
        self.softmax = nn.Softmax(dim=0)
        
        # Loop over all models
        for model_name, model in self.models_dict.items():
            self.stats[model_name] = {}
            if condition_mask is None:
                condition_mask = torch.cat([torch.zeros(model.nodes_size-x.shape[-1]),torch.ones(x.shape[-1])])
            self.condition_mask = condition_mask

            ####################
            # Posterior sampling
            posterior_samples = model.sample(x=x, err=err, condition_mask=condition_mask,
                                            timesteps=timesteps, eps=eps, num_samples=num_samples, cfg_alpha=cfg_alpha,
                                            multi_obs_inference=multi_obs_inference, hierarchy=hierarchy,
                                            order=order, snr=snr, corrector_steps_interval=corrector_steps_interval, corrector_steps=corrector_steps, final_corrector_steps=final_corrector_steps,
                                            device=device, verbose=verbose, method=method)
            posterior_samples = posterior_samples.cpu().numpy()

            # Inference Attention weights
            self.stats[model_name]["attn_weights"] = model.sampler.all_attn_weights

            # MAP estimation
            theta_hat = np.array([self._map_kde(posterior_samples[i]) for i in range(len(posterior_samples))])
            MAP_posterior, std_MAP_posterior = torch.tensor(theta_hat[:,0], dtype=torch.float), torch.tensor(theta_hat[:,1], dtype=torch.float)

            # Storing MAP and std MAP
            self.stats[model_name]["MAP"] = theta_hat

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

        model_print_length = len(max(model_names, key=len))
        print(f"Probabilities of the models after {len(x)} observations:")
        for model in model_names:
            print(f"{model.ljust(model_print_length)}: {100*self.stats[model]['model_prob']:6.2f} %")
        print()
        print(f"Model {best_model} fits the data best " + 
                f"with a relative support of {best_model_prob:.1f}% among the considered models.")
        
        if self.path is not None:
            with open(f"{self.path}/model_comp.pkl", "wb") as f:
                pickle.dump(self.stats, f)

    #############################################
    # ----- Kernel Density Estimation -----
    #############################################

    #---------------------------
    # Estimate the log probability
    def _log_prob(self, samples, observation):
        """Compute the log probability of the samples"""
        kde = gaussian_kde(samples.T)
        log_prob = kde.logpdf(observation).item()
        return log_prob

    #---------------------------
    # Estimate the Maximum A Posteriori (MAP)
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
    # ----- Plotting -----
    ##############################################
    
    #---------------------------
    # Model Comparison plotting
    def plot_comparison(self, stats_dict=None, n_models=10, sort="median", model_names=None, path=None, show=True):
        """
        Plot the results from the Model Comparison.
        Saves the Violin plots for individual model probability and the cumulative model probability of all observations.

        Args:
            stats_dict: (dict)(optional) Dictionary with the comparison results. 
                            If not provided, it uses the results from the `compare()` call.
            n_models:   (int) Number of models to plot in the comparison.
            sort:       (string) How to sort the models for the plots.
                            median - (default) median model probability of all observations
                            mean   -  mean model probability of all observations
                            none   - the order the models are defined in
            model_names: (list of strings)(optional) List with the names of the models to plot.
            path:       (string)(optional) The path the plots are saved to.
                            If not provided, the plots are not saved.
            show:       (bool) Whether to show the created plots or not.
        """

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

        model_keys = list(stats_dict.keys())
        model_probs = torch.tensor([stats_dict[model]["model_prob"] for model in model_keys])
        model_log_probs = torch.stack([stats_dict[model]["log_probs"] for model in model_keys])
        model_obs_probs = torch.stack([stats_dict[model]["obs_probs"] for model in model_keys])
        if model_names is None:
            model_names = model_keys

        if len(model_names) < n_models:
            n_models = len(model_names)

        legend_cols = 1 if len(model_names) < 6 else 2

        # plt.style.use('ggplot')

        #---------------------------
        # Plot violin plot of model probabilities
        plt.figure(figsize=(12, 6), dpi=500)
        model_names_violin = [name.replace(", ", "\n") for name in model_names[:n_models]]
        sns.violinplot(data=model_obs_probs.T[:,:n_models],label=model_names_violin, palette=sns.color_palette("dark"), inner_kws=dict(box_width=5, whis_width=2, color="k"))

        if model_names != "":
            plt.xticks(ticks=range(n_models), labels=model_names_violin)
            plt.tick_params(axis='x', which='major', labelsize=16)

        plt.tick_params(axis='y', which='major', labelsize=16)
        plt.ylabel(r"$P(\mathcal{M} | x_i)$", fontsize=20)
        sns.despine()
        plt.tight_layout()

        if path is not None:
            plt.savefig(f"{path}/model_probs_violin.png")
        if show:
            plt.show()
        plt.close()

        #---------------------------
        # Plot cumulative model probabilities

        # Calculate mean model probabilities for N observations
        avg_model_probs = []
        for n in range(50):
            all_N_log_probs = []
            for i in range(0,model_log_probs.shape[1]+1):
                if i != 0:
                    idx = torch.randperm(model_log_probs.shape[1])[:i]
                    N_log_probs = model_log_probs[:,idx].T
                elif i == 0:
                    N_log_probs = torch.zeros_like(model_log_probs[:,0]).unsqueeze(0)

                all_N_log_probs.append(torch.nn.functional.softmax(N_log_probs.sum(0),0).T)
            all_N_log_probs = torch.stack(all_N_log_probs)
            avg_model_probs.append(all_N_log_probs)

        avg_model_probs = torch.stack(avg_model_probs)
        avg_mean = avg_model_probs.mean(0)
        avg_std = avg_model_probs.std(0)/torch.sqrt(torch.tensor(avg_model_probs.shape[0]))

        plt.figure(figsize=(12, 6), dpi=500)
        palette = sns.color_palette("dark", n_colors=n_models)
        for n in range(n_models):
            plt.errorbar(
            torch.arange(0, model_log_probs.shape[1]+1).T,
            avg_mean[:, n], yerr=avg_std[:, n],
            label=model_names[n], marker='o', markersize=6, linewidth=3, elinewidth=1, capsize=2,
            color=palette[n]
            )
        if model_names != "":
            plt.legend(title="Models", loc='right', fontsize=15, title_fontsize=16, frameon=True, ncol=legend_cols)
    
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel("# Observations", fontsize=20)
        plt.ylabel(r"$P(\mathcal{M} | x_0,..., x_i)$", fontsize=20)
        # plt.grid(True)
        sns.despine()
        plt.tight_layout()
        if path is not None:
            plt.savefig(f"{path}/model_probs_cumulative.png")
        if show:
            plt.show()
        plt.close()

    #---------------------------
    # Attention Heatmap plotting
    def plot_attention(self, stats_dict=None, labels=None, path=None, show=True):
        """
        Plot the attention weights for the best performing model for interpretability.

        Args:
            stats_dict: (dict)(optional) Dictionary with the comparison results. 
                            If not provided, it uses the results from the `compare()` call.
            labels:     (list of strings)(optional) List with the names of the parameters and data points.
            path:       (string)(optional) The path the plots are saved to.
                            If not provided, the plots are not saved.
            show:       (bool) Whether to show the created plots or not.
        """

        # Check path
        if path is None:
            path = self.path
            if not os.path.exists(path):
                os.makedirs(path)

        # Check stats_dict
        if stats_dict is None:
            stats_dict = self.stats

        # Get the best performing model
        best_model = sorted(stats_dict, key=lambda x: stats_dict[x]["AIC"], reverse=True)[0]

        def _plot_heatmap(data, xlabels, ylabels, name, show):
            # Set annotations in the attention blocks
            annotation_mask = data > 0.0
            annot = np.where(annotation_mask, data.round(2), np.nan)  # Use NaN to hide annotations below threshold
            annotations = annot.astype(str)
            annotations[np.isnan(annot)] = ""

            # Set up colours
            vmin, vmax = 0.0, 1.0
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) 

            # Create figure
            fig = plt.figure(figsize=(12,6), dpi=500)
            ax = sns.heatmap(
                data,
                xticklabels=xlabels,
                yticklabels=ylabels,
                cmap='magma',
                cbar=False,
                linewidths=.5,
                square=False,
                vmin=vmin,
                vmax=vmax,
                annot=annotations,
                norm=norm,
                fmt='',
                annot_kws={"size": 35 / np.sqrt(len(data))}
            ) 

            ax.set_xlabel("Keys", fontsize=25)
            ax.set_ylabel("Queries", fontsize=25)
            plt.xticks(rotation=0, ha='center', fontsize=20)
            plt.yticks(rotation=0, fontsize=20)

            # Add the single, shared color bar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]

            # Create the color bar using a "dummy" mappable object
            sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
            sm._A = [] # Dummy empty array
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('Attention Weight', fontsize=20)

            fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust rect to make space for suptitle and cbar

            
            #plt.tight_layout()
            if path is not None:
                plt.savefig(f"{path}/{name}.png")
            if show:
                plt.show()
            plt.close()
        
        ####################
        # Avg Attention between informative Tokens

        data = stats_dict[best_model]["attn_weights"].mean(0).numpy()
        data = data[np.ix_(~self.condition_mask.bool(),torch.cat((self.condition_mask.bool(), torch.tensor([True]))))]

        xlabels = list(compress(labels, self.condition_mask)) + ["Bias KV"]
        ylabels = list(compress(labels, 1-self.condition_mask))
        _plot_heatmap(data, xlabels, ylabels, "selected_attention_map", show)

        ####################
        # Layer by Layer Attention

        data = stats_dict[best_model]["attn_weights"].numpy()

        # Create a list to hold the data for each layer
        plot_data = []
        for layer_weights in data:
            avg_attention_map = layer_weights
            
            param_attention_subset = avg_attention_map[np.ix_(~self.condition_mask.bool(),torch.cat((self.condition_mask.bool(), torch.tensor([True]))))]
            plot_data.append(param_attention_subset)

        # Set up Figure   
        nrows = stats_dict[best_model]["attn_weights"].shape[2]
        fig, axes = plt.subplots(
            nrows=nrows, 
            ncols=1, 
            figsize=(12, 3*nrows),
            sharex=True,
            dpi=500
        )

        # Set up colours
        vmin, vmax = 0.0, 1.0
        norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax) 

        # Loop through each layer and plot the heatmap
        for i, ax in enumerate(axes):
            # Get the data for the current layer
            data_to_plot = plot_data[i]

            # Create a boolean mask for annotations. Only show values > 0.1
            annotation_mask = data_to_plot > 0.1
            annot = np.where(annotation_mask, data_to_plot.round(2), np.nan)  # Use NaN to hide annotations below threshold
            annotations = annot.astype(str)
            annotations[np.isnan(annot)] = ""
            
            # Create the heatmap on the current subplot axis `ax`
            sns.heatmap(
                data_to_plot,
                xticklabels=xlabels,
                yticklabels=ylabels,
                cmap='magma',
                linewidths=.5,
                ax=ax,
                cbar=False,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                # annot=annotations,
                fmt="",
                # annot_kws={"size": 35 / np.sqrt(len(data_to_plot))}
            )
            
            # Set titles and labels for each subplot
            ax.set_title(f"Layer {i+1}", fontsize=25, loc='left')
            ax.tick_params(axis='y', rotation=0, labelsize=20) # Rotate y-axis labels for better readability

            # Only show x-axis labels on the very last plot
            if i == len(axes) - 1:
                #ax.set_xlabel("Information Source (Observations and Bias)", fontsize=14)
                ax.tick_params(axis='x', rotation=0, labelsize=20)
            else:
                ax.set_xlabel('')

        # Add the single, shared color bar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]

        # Create the color bar using a "dummy" mappable object
        sm = plt.cm.ScalarMappable(cmap='magma', norm=norm)
        sm._A = [] # Dummy empty array
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Attention Weight', fontsize=20)

        fig.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust rect to make space for suptitle and cbar

        if path is not None:
            plt.savefig(f"{path}/layer_attention.png")
        if show:
            plt.show()
        plt.close()
