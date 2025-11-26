# SMDPfier

[![PyPI Version](https://img.shields.io/pypi/v/smdpfier.svg)](https://pypi.org/project/smdpfier/)
[![Python Version](https://img.shields.io/pypi/pyversions/smdpfier.svg)](https://pypi.org/project/smdpfier/)
[![License](https://img.shields.io/pypi/l/smdpfier.svg)](https://github.com/smdpfier/smdpfier/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/smdpfier/smdpfier/CI)](https://github.com/smdpfier/smdpfier/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://smdpfier.readthedocs.io)
[![Coverage](https://img.shields.io/codecov/c/github/smdpfier/smdpfier.svg)](https://codecov.io/gh/smdpfier/smdpfier)

**Add SMDP-level behavior to any Gymnasium environment with options and simple temporal semantics.**

SMDPfier is a Gymnasium wrapper that enables Semi-Markov Decision Process (SMDP) behavior by letting you execute **Options** (sequences of primitive actions) where each primitive action = 1 tick of time, enabling natural SMDP discounting with γ^{k}.

## 🚀 Quick Start

### Index Interface (Recommended for RL)

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option

# Create environment and define options
env = gym.make("CartPole-v1")
options = [
    Option(actions=[0, 0, 1], name="left-left-right"),     # 3 actions = 3 ticks
    Option(actions=[1, 1, 0], name="right-right-left"),    # 3 actions = 3 ticks
    Option(actions=[0, 1], name="left-right"),              # 2 actions = 2 ticks
]

# Wrap with SMDPfier
smdp_env = SMDPfier(
    env,
    options_provider=options,           # Static options list
    action_interface="index",           # Discrete(3) action space
    max_options=len(options)
)

# Use it like any Gym environment
obs, info = smdp_env.reset()
obs, reward, term, trunc, info = smdp_env.step(0)  # Execute first option

# Access SMDP metadata
smdp_info = info["smdp"]
print(f"Option: {smdp_info['option']['name']}")
print(f"Duration: {smdp_info['duration']} ticks (= k_exec)")
print(f"Per-step rewards: {smdp_info['rewards']}")
print(f"Macro reward: {reward}")  # sum of per-step rewards

# Apply SMDP discounting: γ^{duration}
gamma = 0.99
discounted_reward = reward * (gamma ** smdp_info['duration'])
```

### Direct Interface (Intuitive)

```python
# Pass Option objects directly
smdp_env = SMDPfier(env, options_provider=options, action_interface="direct")

# Execute with Option objects
obs, reward, term, trunc, info = smdp_env.step(options[0])
```

## 🎯 Key Features

- **⏱️ Simple Time Semantics**: Each primitive action = 1 tick, duration = k_exec
- **🔗 Flexible Options**: Static sequences or dynamic discovery via callable
- **🎛️ Two Interfaces**: Index-based (`Discrete` actions) or direct `Option` passing
- **📊 SMDP Discounting**: Natural γ^{k} discounting where k = number of primitive actions
- **🎭 Action Masking**: Support for discrete action availability constraints
- **📋 Rich Info**: Comprehensive execution metadata in `info["smdp"]`
- **🛡️ Error Handling**: Detailed validation and runtime error reporting
- **🔄 Continuous Actions**: Full support for continuous action spaces
- **🎲 Built-in Defaults**: Ready-to-use option generators and reward aggregators

## 📖 Core Concepts

### Options
**Options** are sequences of primitive actions executed atomically:
```python
# Simple option with 3 primitive actions
option = Option(actions=[0, 1, 0], name="left-right-left")
```

### Time Semantics (v0.2.0+)
**Simple and natural:**
- Each primitive action = **1 tick** of time
- Option duration = **k_exec** (number of primitive actions executed)
- If option completes: duration = `len(option.actions)`
- If terminated early: duration < `len(option.actions)`

```python
# This option always executes 3 steps = 3 ticks
option = Option(actions=[0, 1, 0], name="three-steps")

# If it completes: duration = 3 ticks
# If terminated after 2 steps: duration = 2 ticks
```

### SMDP Discounting
Standard MDP: `γ^{1}` per step | SMDP: `γ^{k}` where k = option duration
```python
# Standard MDP discounting (each primitive step)
mdp_value = r1 + γ¹·r2 + γ²·r3 + γ³·r4

# SMDP discounting with options of lengths [3, 2, 4]
smdp_value = r1 + γ³·r2 + γ⁵·r3 + γ⁹·r4
#                   ↑      ↑       ↑
#                   3    3+2    3+2+4
```

## 🔧 Interfaces

### Index Interface (Recommended for RL)
```python
# Exposes Discrete(max_options) action space
smdp_env = SMDPfier(
    env,
    options_provider=options,
    action_interface="index",
    max_options=len(options)
)

# Use integer indices
action = 1  # Select second option
obs, reward, term, trunc, info = smdp_env.step(action)
```

### Direct Interface (Intuitive)
```python
# Pass Option objects directly
smdp_env = SMDPfier(
    env, 
    options_provider=options,
    action_interface="direct"
)

# Use Option objects
option = options[1]
obs, reward, term, trunc, info = smdp_env.step(option)
```

## 📚 Documentation

| Topic | Description |
|-------|-------------|
| [**API Reference**](docs/api.md) | Complete API documentation and examples |
| [**Durations Guide**](docs/durations.md) | Understanding duration = k_exec and SMDP discounting |
| [**Index vs Direct**](docs/usage_index_vs_direct.md) | Choosing the right interface for your use case |
| [**Masking & Precheck**](docs/masking_and_precheck.md) | Action constraints and validation |
| [**Error Handling**](docs/errors.md) | Comprehensive error context and debugging |
| [**FAQ**](docs/faq.md) | Common questions and gotchas |
| [**Migration from 0.1.x**](docs/migration_0_2.md) | Upgrading to v0.2.0 simplified semantics |

## 🔍 SMDP Info Payload

Every step returns rich metadata in `info["smdp"]`:

```python
{
    "option": {
        "id": "abc123...",           # Stable hash-based ID
        "name": "left-right-left",   # Human-readable name
        "len": 3,                    # Number of primitive actions  
        "meta": {}                   # User metadata
    },
    "k_exec": 3,                     # Primitive steps executed
    "duration": 3,                   # Duration in ticks (= k_exec)
    "rewards": [1.0, 1.0, 1.0],     # Per-step rewards
    "terminated_early": False,       # Whether episode ended during option
    "action_mask": [1, 1, 0],       # Available option indices (index interface)
    "num_dropped": 0                 # Options dropped due to overflow (index interface)
}
```

## 🎲 Built-in Defaults

### Option Generators
```python
from smdpfier.defaults.options import RandomStaticLen, RandomVarLen

# Fixed-length random options
RandomStaticLen(length=3, action_space_size=4, num_options=10)

# Variable-length random options  
RandomVarLen(min_length=2, max_length=5, action_space_size=4, num_options=8)
```

### Reward Aggregation
```python
from smdpfier.defaults import sum_rewards, mean_rewards, discounted_sum

# Sum all per-step rewards (default)
reward_agg=sum_rewards

# Average per-step rewards
reward_agg=mean_rewards

# Discount per-step rewards with γ=0.99
reward_agg=discounted_sum(gamma=0.99)
```

## 🔧 Installation

```bash
pip install smdpfier
```

**Development Install:**
```bash
git clone https://github.com/smdpfier/smdpfier.git
cd smdpfier
pip install -e ".[dev,docs]"
```

## 🧪 Examples

See the [`examples/`](examples/) directory:
- [`cartpole_index_static.py`](examples/cartpole_index_static.py) - Index interface with static options
- [`pendulum_direct_continuous.py`](examples/pendulum_direct_continuous.py) - Direct interface with continuous actions
- [`taxi_index_dynamic_mask.py`](examples/taxi_index_dynamic_mask.py) - Dynamic options with action masking

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 Citation

If you use SMDPfier in your research, please cite:

```bibtex
@software{smdpfier2024,
  title = {SMDPfier: SMDP-level behavior for Gymnasium environments},
  author = {Erel A. Shtossel, Gal A. Kaminka},
  url = {https://github.com/smdpfier/smdpfier},
  year = {2024}
}
```

---

**[📖 Documentation](https://smdpfier.readthedocs.io) | [🐛 Issues](https://github.com/smdpfier/smdpfier/issues) | [💬 Discussions](https://github.com/smdpfier/smdpfier/discussions)**
