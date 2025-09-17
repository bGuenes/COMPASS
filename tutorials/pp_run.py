from autocvd import autocvd
autocvd(num_gpus=9, interval=1)

from src.compass import ModelTransfuser as MTf
import torch


# -----------------------------------------------------------
# --- 1. General ODE Solver (using simple Euler method) ---
def solve_ode(model_func, initial_state, params, t_max, dt):
    """
    Solves a system of ODEs using the Euler method.

    Args:
        model_func (function): The function defining the ODEs (e.g., lotka_volterra).
        initial_state (torch.Tensor): The starting values [N, P].
        params (dict): A dictionary of parameters for the model.
        t_max (int): The maximum simulation time.
        dt (float): The time step.

    Returns:
        (torch.Tensor, torch.Tensor): Tensors for time points and population history.
    """
    # Setup time steps and history arrays
    time_steps = torch.arange(0, t_max, dt)
    history = torch.zeros(initial_state.shape[0], len(time_steps), 2)
    history[:, 0, :] = initial_state
    
    # Current state starts at the initial state
    current_state = initial_state.clone()

    # Euler integration loop
    for i in range(1, len(time_steps)):
        # Get the derivatives from the model function
        derivatives = model_func(current_state, params)
        # Update the state using the Euler step
        current_state += derivatives * dt
        # Ensure populations don't go below zero
        current_state = torch.max(current_state, torch.tensor([0.0, 0.0]))
        history[:, i, :] = current_state
        
    return time_steps, history

# -----------------------------------------------------------
# --- 2. The Four Competing Model Functions ---
# Each model uses exactly four parameters: {alpha, beta, gamma, delta}

def lotka_volterra(state, params):
    """Model 1: Classic Lotka-Volterra dynamics."""
    N, P = state.T
    alpha, beta, gamma, delta = params.T

    dN_dt = alpha * N - beta * N * P
    dP_dt = delta * N * P - gamma * P
    return torch.stack([dN_dt, dP_dt]).T

def logistic_prey(state, params):
    """Model 2: Prey with logistic growth."""
    
    N, P = state.T
    alpha, beta, gamma, delta = params.T
    delta_lp = delta * 1000  # Prey carrying capacity (logistic growth)
    cn_rate = 0.5  # Fixed conversion efficiency

    dN_dt = alpha * N * (1 - N / delta_lp) - beta * N * P
    dP_dt = cn_rate * beta * N * P - gamma * P
    return torch.stack([dN_dt, dP_dt]).T

def satiated_predator(state, params):
    """Model 3: Predator with satiation (Holling Type II)."""
    N, P = state.T
    alpha, beta, gamma, delta = params.T
    cn_rate = 0.5 # Fixed conversion efficiency

    consumption = (beta * N) / (1 + beta * delta * N)
    dN_dt = alpha * N - consumption * P
    dP_dt = cn_rate * consumption * P - gamma * P
    return torch.stack([dN_dt, dP_dt]).T

def rosenzweig_macarthur(state, params):
    """Model 4: Both logistic prey and satiated predator."""
    N, P = state.T
    alpha, beta, gamma, delta = params.T
    delta_rm = delta * 1000  # 
    cn_rate = 0.5 # Fixed conversion efficiency
    h_rate = 0.1 # Fixed handling time

    consumption = (beta * N) / (1 + beta * h_rate * N)
    dN_dt = alpha * N * (1 - N / delta_rm) - consumption * P
    dP_dt = cn_rate * consumption * P - gamma * P
    return torch.stack([dN_dt, dP_dt]).T

# -----------------------------------------------------------
# --- 4. Run Simulations and Collect Data ---
class prior_distributions:
    def __init__(self):
        self.alpha = torch.distributions.normal.Normal(-0.125, 0.5)
        self.beta = torch.distributions.normal.Normal(-3, 0.5)
        self.gamma = torch.distributions.normal.Normal(-0.125, 0.5)
        self.delta = torch.distributions.normal.Normal(-3, 0.5)

    def sample(self, num_samples=1):
        alpha = self.alpha.sample((num_samples,))
        beta = self.beta.sample((num_samples,))
        gamma = self.gamma.sample((num_samples,))
        delta = self.delta.sample((num_samples,))

        params = torch.stack([alpha, beta, gamma, delta], dim=-1)

        return params

if __name__ == "__main__":
    print("Running...")
    # List of models to run
    models = {
    "Lotka-Volterra": lotka_volterra,
    "Logistic Prey": logistic_prey,
    "Satiated Predator": satiated_predator,
    "Rosenzweig-MacArthur": rosenzweig_macarthur
    }


    prior = prior_distributions()

    training_data = {}
    validation_data = {}
    noise_fn = torch.distributions.normal.Normal(0, 0.1)

    for model_name, model_func in models.items():
        # Sample initial conditions and parameters
        init_state = torch.tensor([[30.0, 1.0]])
        log_init_state = torch.log(init_state)

        log_params = prior.sample(500_000)
        log_val_params = prior.sample(50_000)

        params = torch.exp(log_params)
        val_params = torch.exp(log_val_params)

        t_max = 20
        dt = 0.01

        # -----------------
        # Training Data
        init_state_train = init_state.repeat([params.shape[0], 1])
        time, history = solve_ode(model_func, init_state_train, params, t_max, dt)

        # get full timestep data
        data = history[:, time % 1 == 0]
        data = data.flatten(1)

        # Remove any NaN values
        log_params = log_params[~torch.any(data.isnan(), dim=1)]
        data = data[~torch.any(data.isnan(), dim=1)]

        # add log normal noise
        noise = noise_fn.sample(data.shape)
        data = data + noise
        data = data/100  # scale down to avoid large values

        data = torch.clamp(data, min=10**(-12))

        # Store the training data
        training_data[model_name] = {
            "theta": log_params,
            "x": data
        }

        # -----------------
        # Validation Data
        init_state_val = init_state.repeat([val_params.shape[0], 1])
        time, history = solve_ode(model_func, init_state_val, val_params, t_max, dt)

        # get full timestep data
        data = history[:, time % 1 == 0]
        data = data.flatten(1)

        # Remove any NaN values
        log_val_params = log_val_params[~torch.any(data.isnan(), dim=1)]
        data = data[~torch.any(data.isnan(), dim=1)]

        # add log normal noise
        noise = noise_fn.sample(data.shape)
        data = data + noise
        data = data/100  # scale down to avoid large values

        data = torch.clamp(data, min=10**(-12))

        # Store the validation data
        validation_data[model_name] = {
            "theta": log_val_params,
            "x": data
        }

    print("Data generation complete.")

    # -----------------------------------------------------------
    mtf = MTf(path="data/predator_prey")

    for model_name, model_func in models.items():
        theta = training_data[model_name]["theta"]
        x = training_data[model_name]["x"]
        val_theta = validation_data[model_name]["theta"]
        val_x = validation_data[model_name]["x"]
        
        mtf.add_data(model_name, theta, x, val_theta, val_x)

    mtf.init_models(sde_type='vesde', sigma=2, hidden_size=50, depth=8, num_heads=5, mlp_ratio=4)
    mtf.train_models(batch_size=1000, verbose=False)

    print("Finished!")
