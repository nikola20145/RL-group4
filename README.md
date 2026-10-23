# Maze Navigation with Q-Learning

A reinforcement learning group project: an agent learns to navigate a 10×10 maze using **tabular Q-learning**, built on a custom [Gymnasium](https://gymnasium.farama.org/) environment. We compare a random policy, a greedy (deterministic) learned policy and an ε-greedy (stochastic) learned policy.

## The environment

`MazeEnv` is a custom `gymnasium.Env`:

```
█ █ █ █ █ █ █ █ █ █
█ S █ . . . █ . . █
█ . █ . █ . █ . █ █
█ . . . █ . . . . █
█ █ █ . █ █ █ █ . █
█ . █ G . . . █ . █
█ . . . █ █ . . . █
█ █ █ . █ . █ █ . █
█ . . . . . █ E . █
█ █ █ █ █ █ █ █ █ █
```

| Element | Description |
|---|---|
| **S** – start | Agent starts at `(1, 1)` |
| **G** – sub-goal | `(5, 3)`, gives a small bonus reward |
| **E** – end goal | `(8, 7)`, ends the episode |
| **Actions** | 4 discrete: up, down, left, right (moves into walls are ignored) |
| **States** | 100 discrete states (each grid cell flattened to an index) |

**Rewards:** −1 per step, +2 for stepping on the sub-goal, +100 for reaching the goal.

## Method

The Q-table (100 states × 4 actions) is updated with the standard Q-learning rule:

```
Q(s, a) ← (1 − α) · Q(s, a) + α · (r + γ · max_a' Q(s', a'))
```

with learning rate **α = 0.1** and discount factor **γ = 0.8**, trained for 10,000 episodes. The learned policy is then evaluated over 10,000 episodes in two modes:

- **Deterministic:** always take `argmax Q(s, ·)`
- **Stochastic (ε-greedy):** take a random action with probability ε = 0.1

## Results

| Policy | Average reward per episode |
|---|---|
| Random actions (during training) | −778.5 |
| Learned policy, stochastic (ε = 0.1) | 87.6 |
| **Learned policy, deterministic** | **89.0** |

The learned policy finds a short path to the goal: it goes from about −779 average reward with random actions to 89. The deterministic policy beats the ε-greedy one slightly, as we expected, because random exploratory moves at evaluation time only add extra steps.

### Other things we looked at
- How the sub-goal reward value changes whether the learned path detours through the sub-goal
- How α and γ affect learning performance

## Repository contents

| File | Description |
|---|---|
| `project.ipynb` | Main notebook: environment, Q-learning training, evaluation and comparison plot |
| `stochastic_deter_Q-learning.ipynb` | Variant that trains with an ε-greedy policy and random tie-breaking for unvisited states |
| `rl-group4.py` | Early standalone version of the maze environment with random-agent visualisation |

## Running it

```bash
pip install gymnasium numpy matplotlib seaborn jupyter
jupyter notebook project.ipynb
```

## Team

Group 4 – [@nikola20145](https://github.com/nikola20145), [@gaellenehme](https://github.com/gaellenehme), [@AryanKheskani](https://github.com/AryanKheskani), [@Abdelrahmanmemara](https://github.com/Abdelrahmanmemara)
