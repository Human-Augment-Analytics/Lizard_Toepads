1. The Problem

Each YOLO training run takes hours on GT PACE cluster
Grid search would need 100+ experiments = weeks of compute time
Need a smarter approach

2. Core Idea: Probabilistic Model

Traditional methods treat each trial independently
Bayesian optimization builds a surrogate model (usually Gaussian Process) that learns: hyperparameters → model performance
Key: it predicts not just a value, but a probability distribution (mean + uncertainty)

3. How It Decides Next Trial

After each experiment, updates the probabilistic model
Acquisition function (like Expected Improvement) uses:

Mean: exploit known good regions
Variance/uncertainty: explore under-sampled areas


Picks the hyperparameter config with highest "potential value"

4. The Advantage

Learns from ALL previous trials
Intelligently balances exploration vs exploitation
~20-30 trials instead of 100+ to find optimal configs

5. Implementation with Ray Tune

Ray Tune handles distributed execution on PACE
Built-in Bayesian search algorithms (I'm using Optuna backend)
Easy to parallelize across multiple GPUs