<p align="center">
    <img src="https://raw.github.com/bGuenes/COMPASS/main/docs/COMPASS_logo.png" width="50%">
</p>

[![PyPi version](https://img.shields.io/pypi/v/bayes-compass.svg)](https://pypi.org/project/bayes-compass/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15183599.svg)](https://doi.org/10.5281/zenodo.15183599)
![Static Badge](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Static Badge](https://img.shields.io/badge/License-GPLv3-yellow.svg)
![Static Badge](https://img.shields.io/badge/Status-Active-green.svg)

# COMPASS: Comparison Of Models using Probabilistic Assessment in Simulation-based Settings
`COMPASS` is a Python package designed for Bayesian Model Comparison in simulation-based settings. By comparing the predictive power of various models, it aims to identify the most suitable model for a given dataset. <br>
It is especially suited for fields like astrophysics and computational biology, where simulation is integral to the modeling process.


# Features
- Perform Bayesian model comparison in simulation-based settings with `ModelTransfuser`
- Perform Simulation-Based Inference with `ScoreBasedInferenceModel`


# Installation
Install the package using pip:
```bash
pip install bayes-compass
```


# Usage
## Model Comparison Example
The `ModelTransfuser` class provides a framework for the model comparison workflow. <br>
It uses the `ScoreBasedInferenceModel` class to perform the simulation-based inference, predicts the posterior distribution and samples from the Likelihood function with the inferred parameters. <br>
With a Gaussian Kernel Density Estimator, it evaluates the observed data at the Likelihood function and computes the posterior model probabilities. <br>
```python
from compass import ModelTransfuser 

# Initialize the ModelTransfuser
mtf = ModelTransfuser()

# Add data from simulators
mtf.add_data(model_name="Model1", train_data=data_1, val_data=val_data_1)
mtf.add_data(model_name="Model2", train_data=data_2, val_data=val_data_2)

# Initialize ScoreBasedInferenceModels
mtf.init_models()

# Train the models
mtf.train_models()

# Compare Posterior Model Probabilities
observations = load_your_observations
mtf.compare(x=observations, err=observations_err)

stats = mtf.stats

# Plot results
mtf.plots()
```

`mtf.stats` is a dictionary containing all inferred parameters and their uncertainties, the posterior model probabilities, and the log-likelihood values for each model. <br>

## Simulation-Based Inference Model
The `ScoreBasedInferenceModel` is a Diffusion Model with a score predicting Transformer network. <br>
It is able to sample from the posterior and likelihood function by utalizing the attention mechanism of the Transformer architecture. <br>
The model is also able to handle observational uncertainties and is designed to run on all available GPUs. <br>
```python
from compass import ScoreBasedInferenceModel as SBIm

sbimodel = SBIm(nodes_size=nodes_size, sigma=sigma, depth=depth, hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)

sbimodel.train(train_data, val_data=val_data, path=path, device="cuda")
```

`nodes_size` - The number of parameters $\theta$ and the number of data points $x$. <br>
`sigma` - The noise level of the initial noise distribution of the diffusion model. <br>
`depth` - The number of layers in the Transformer <br>
`hidden_size` - The size of the embedding in the Transformer <br>
`num_heads` - The number of attention heads in the Transformer <br>
`mlp_ratio` - The ratio of the hidden size to the number of Nodes in the MLPs in the Transformer  <br>

To sample from the posterior distribution, you can use the `sample()` function and provide the observations `x` and the observational uncertainties `err` (optional):
```python
# Sample from the posterior distribution
posterior_samples = sbimodel.sample(x=observations, err=observations_err, timesteps=100)
```
To sample from the Likelihood function, you can again use the `sample()` function and provide the Maximum-A-Posteriori $\hat\theta$ values in `theta` and the standard deviation `err` (optional):
```python
# Samlpe from the likelihood function
likelihood_samples = sbimodel.sample_likelihood(theta=theta_hat, err=std_theta_hat, timesteps=100)
```


# Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this package.

---