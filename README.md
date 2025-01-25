# DDPG and TD3 Algorithm Replication

## Project Description
This project is a research implementation of two important reinforcement learning algorithms: 
- Deep Deterministic Policy Gradient (DDPG)
- Twin Delayed DDPG (TD3)

The goal is to reproduce the key findings and performance of the original papers through faithful algorithmic implementation.

## Papers Replicated
- DDPG: "Continuous control with deep reinforcement learning" (Lillicrap et al., 2015)
- TD3: "Addressing Function Approximation Error in Actor-Critic Methods" (Fujimoto et al., 2018)

## Installation

### Prerequisites
- Python 3.8+
- Poetry for dependency management

### Dependencies
This project uses `pyproject.toml` for dependency management. To install:

```bash
poetry install
```


## Algorithms Overview

### DDPG
- **Type**: Actor-Critic, Off-Policy
- **Key Features**:
  - Continuous action spaces
  - Deterministic policy gradient
  - Experience replay
  - Target networks

### TD3
- **Type**: Actor-Critic, Off-Policy
- **Improvements over DDPG**:
  - Clipped double Q-learning
  - Delayed policy updates
  - Target policy smoothing


## Contributions
Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## License
MIT License

## References
1. Lillicrap et al. (2015). Continuous control with deep reinforcement learning.
2. Fujimoto et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods.
