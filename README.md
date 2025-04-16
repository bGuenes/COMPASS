<p align="center">
    <img src="docs/COMPASS_logo.png" width="50%">
</p>

![Static Badge](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Static Badge](https://img.shields.io/badge/License-GPLv3-yellow.svg)
![Static Badge](https://img.shields.io/badge/Status-Active-green.svg)

### Comparison Of Models using Probabilistic Assessment in Simulation-based Settings <br>
---
``compass`` is a Python package to perform Bayesian Model Comparison in simulation-based settings, by comparing the predictive power of different models from approximating the likelihood function and also providing the correspondig parameters for the model. <br> 
It is designed to be used in conjunction with simulation models, such as those used in astrophysics or computational biology and other fields where simulation is a key component of the modeling process.

### Installation
```bash
pip install bayes-compass
```

### Comparison of Models
The ``ModelTransfuser`` class provides a framework for the model comparison workflow. It can store the data from different models, train the models, and compare the posterior model probabilities.
```python
from compass import ModelTransfuser

# Initialize the ModelTransfuser
MTf = ModelTransfuser()

# Add the data form the simulators
MTf.add_data(model_name="Model1", train_data=data_1, val_data=val_data_1)
MTf.add_data(model_name="Model2", train_data=data_2, val_data=val_data_2)

# Initialize the ScoreBasedInferenceModels
MTf.init_models()

# Train the models
MTf.train_models()

# Compare the Posterior Model Probabilities
observations = load_your_observations # Load in your observational data
# Set the condition mask to 1 for the observed data indices and 0 for the latent values that will be inferred
condition_mask = specify_condition_mask
MTf.compare(observations, condition_mask)

stats = MTf.stats

# Plot the results
MTf.plots()
```

### Simulation-Based Inference
``compass`` also provides tools for simulation-based inference, allowing for the estimation of parameters.<br>
The ``ScoreBasedInferenceModel`` class is used to perform inference on the models using a score-based approach.
```python
from compass import ScoreBasedInferenceModel

SBIm = ScoreBasedInferenceModel(node_size=)
```

