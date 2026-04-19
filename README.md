# Lab 6 - Reinforcement Learning  
Policy Gradient Methods (REINFORCE)

The lab includes:

### 1. Policy Gradient (REINFORCE) Implementation
- Implements a neural network policy trained using the **REINFORCE algorithm**
- Uses episode-based learning with Monte Carlo returns
- Supports training via a centralized `run_experiment()` function
- Tracks training performance across episodes

### 2. Training Pipeline and Experimentation
- Multiple training runs controlled via `run_experiment`
- Evaluation of learning stability and convergence behavior
- Comparison of reward trajectories over time
- Model checkpointing using `torch.save()`

### 3. Analysis of Policy Behavior
- Monitors:
  - Episode rewards
  - Loss trends
  - Policy stability
- Investigates instability issues such as:
  - high variance gradients
  - sparse reward learning
  - convergence sensitivity to learning rate

---

## Setup

### 1. Install Dependencies

```python
%pip install numpy matplotlib torch gymnasium
