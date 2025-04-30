<p align="center">
    <img src="docs/COMPASS_logo.png" width="50%">
</p>

[![PyPi version](https://img.shields.io/pypi/v/bayes-compass.svg)](https://pypi.org/project/bayes-compass/)
![Static Badge](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Static Badge](https://img.shields.io/badge/License-GPLv3-yellow.svg)
![Static Badge](https://img.shields.io/badge/Status-Active-green.svg)

## COMPASS: Comparison Of Models using Probabilistic Assessment in Simulation-based Settings
`COMPASS` is a Python package designed for Bayesian Model Comparison in simulation-based settings. By comparing the predictive power of various models, it aims to identify the most suitable model for a given dataset. <br>
It is especially suited for fields like astrophysics and computational biology, where simulation is integral to the modeling process.

---

## Features
- Perform Bayesian model comparison in simulation-based settings.
- Simulate, train, and evaluate models with ease.
- Tools for posterior model probability comparison.
- Includes `ModelTransfuser` and `ScoreBasedInferenceModel` classes for seamless workflows.

---

## Installation
Install the package using pip:
```bash
pip install bayes-compass
```

---

## Usage
### Model Comparison Example
The `ModelTransfuser` class provides a framework for model comparison workflows:
```python
from compass import ModelTransfuser

# Initialize the ModelTransfuser
MTf = ModelTransfuser()

# Add data from simulators
MTf.add_data(model_name="Model1", train_data=data_1, val_data=val_data_1)
MTf.add_data(model_name="Model2", train_data=data_2, val_data=val_data_2)

# Initialize ScoreBasedInferenceModels
MTf.init_models()

# Train the models
MTf.train_models()

# Compare Posterior Model Probabilities
observations = load_your_observations
condition_mask = specify_condition_mask
MTf.compare(observations, condition_mask)

stats = MTf.stats

# Plot results
MTf.plots()
```

### Simulation-Based Inference Example
The `ScoreBasedInferenceModel` class allows for estimating parameters using a score-based approach:
```python
from compass import ScoreBasedInferenceModel

SBIm = ScoreBasedInferenceModel(node_size=128)
```

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve this package.

---