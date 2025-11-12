# SMDPfier

[![PyPI Version](https://img.shields.io/pypi/v/smdpfier.svg)](https://pypi.org/project/smdpfier/)
[![Python Version](https://img.shields.io/pypi/pyversions/smdpfier.svg)](https://pypi.org/project/smdpfier/)
[![License](https://img.shields.io/pypi/l/smdpfier.svg)](https://github.com/smdpfier/smdpfier/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/workflow/status/smdpfier/smdpfier/CI)](https://github.com/smdpfier/smdpfier/actions)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://smdpfier.readthedocs.io)
[![Coverage](https://img.shields.io/codecov/c/github/smdpfier/smdpfier.svg)](https://codecov.io/gh/smdpfier/smdpfier)

**Add SMDP-level behavior to any Gymnasium environment with options and proper temporal discounting.**

SMDPfier is a Gymnasium wrapper that enables Semi-Markov Decision Process (SMDP) behavior by letting you execute **Options** (sequences of primitive actions) while tracking **duration metadata** in abstract "ticks" for true SMDP discounting with γ^{ticks}.

## 🚀 Quick Start

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration

# Create environment and define options
env = gym.make("CartPole-v1")
options = [
    Option(actions=[0, 0, 1], name="left-left-right"),
    Option(actions=[1, 1, 0], name="right-right-left"), 
    Option(actions=[0, 1, 0], name="left-right-left"),
]

# Wrap with SMDPfier 
smdp_env = SMDPfier(
    env,
    options_provider=options,           # Static options list
    duration_fn=ConstantOptionDuration(10),  # 10 ticks per option
    action_interface="index"            # Discrete(3) action space
)

# Use it like any Gym environment
obs, info = smdp_env.reset()
obs, reward, term, trunc, info = smdp_env.step(0)  # Execute first option

# Access SMDP metadata
smdp_info = info["smdp"]
print(f"Executed {smdp_info['k_exec']} steps in {smdp_info['duration_exec']} ticks")
print(f"Per-step rewards: {smdp_info['rewards']}")

# Apply SMDP discounting: γ^{duration_exec}
gamma = 0.99
discounted_reward = reward * (gamma ** smdp_info['duration_exec'])
```

## 🎯 Key Features

- **🔗 Flexible Options**: Static sequences or dynamic discovery via callable
- **⚡ Two Interfaces**: Index-based (`Discrete` actions) or direct `Option` passing
- **⏱️ Duration Metadata**: Integer ticks for true SMDP discounting (γ^{ticks})
- **🎭 Action Masking**: Support for discrete action availability constraints
- **📊 Rich Info**: Comprehensive execution metadata in `info["smdp"]`
- **🛡️ Error Handling**: Detailed validation and runtime error reporting
- **🔄 Continuous Actions**: Full support for continuous action spaces
- **🎲 Built-in Defaults**: Ready-to-use option generators and duration functions

## 📖 Core Concepts

### Options
**Options** are sequences of primitive actions executed atomically:
```python
# Simple option with 3 primitive actions
option = Option(actions=[0, 1, 0], name="left-right-left")
```

### Duration (Ticks) vs Steps
**Critical distinction:**
- **Steps**: Number of `env.step()` calls (determined by option length)
- **Duration (Ticks)**: Abstract time units for SMDP discounting (metadata only)

```python
# This option always executes exactly 3 steps
option = Option(actions=[0, 1, 0], name="three-steps")

# But can have any duration (e.g., 10 ticks)
duration_fn = ConstantOptionDuration(10)

# Result: 3 environment steps, 10 ticks of abstract time
```

### SMDP Discounting
Standard MDP: `γ^{steps}` | SMDP: `γ^{ticks}`
```python
# Standard MDP discounting
mdp_value = r1 + γ¹·r2 + γ²·r3 + γ³·r4

# SMDP discounting with option durations [5, 3, 7] ticks  
smdp_value = r1 + γ⁵·r2 + γ⁸·r3 + γ¹⁵·r4
#                   ↑      ↑       ↑
#                   5    5+3    5+3+7
```

## 🔧 Interfaces

### Index Interface (Recommended for RL)
```python
# Exposes Discrete(max_options) action space
smdp_env = SMDPfier(
    env,
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
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
    duration_fn=ConstantOptionDuration(5),
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
| [**Durations Guide**](docs/durations.md) | Understanding ticks, SMDP discounting, and duration policies |
| [**Index vs Direct**](docs/usage_index_vs_direct.md) | Choosing the right interface for your use case |
| [**Masking & Precheck**](docs/masking_and_precheck.md) | Action constraints and validation |
| [**Error Handling**](docs/errors.md) | Comprehensive error context and debugging |
| [**FAQ**](docs/faq.md) | Common questions and gotchas |

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
    "rewards": [1.0, 1.0, 1.0],     # Per-step rewards
    "duration_planned": 10,          # Expected ticks
    "duration_exec": 10,             # Actual ticks (accounts for early termination)
    "terminated_early": False,       # Whether episode ended during option
    "time_units": "ticks",           # Always "ticks"
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

### Duration Functions
```python
from smdpfier.defaults.durations import (
    ConstantOptionDuration,    # Same duration per option
    RandomOptionDuration,      # Random duration per option
    ConstantActionDuration,    # Same duration per action
    RandomActionDuration,      # Random duration per action
    MapActionDuration          # Map actions to durations
)

# 10 ticks per option
ConstantOptionDuration(10)

# 2-5 ticks per action
RandomActionDuration(min_duration=2, max_duration=5)
```

### Reward Aggregation
```python
from smdpfier.defaults.rewards import sum_rewards, mean_rewards, discounted_sum

# Sum all per-step rewards (default)
reward_agg=sum_rewards

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
  author = {SMDPfier Contributors},
  url = {https://github.com/smdpfier/smdpfier},
  year = {2024}
}
```

---

**[📖 Documentation](https://smdpfier.readthedocs.io) | [🐛 Issues](https://github.com/smdpfier/smdpfier/issues) | [💬 Discussions](https://github.com/smdpfier/smdpfier/discussions)**
