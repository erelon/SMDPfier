# SMDPfier Documentation

Welcome to SMDPfier, a Gymnasium wrapper that enables **Semi-Markov Decision Process (SMDP)** behavior in reinforcement learning environments through **Options** with simple, natural time semantics.

## Overview

SMDPfier transforms any Gymnasium environment into an SMDP by allowing agents to execute **Options** (sequences of primitive actions) where each primitive action = 1 tick of time, enabling natural SMDP discounting.

**🎯 Key Insight**: Each primitive action = 1 tick. Option duration = number of actions executed. Simple and natural.

## Key Features

- **🔗 Flexible Options**: Static sequences or dynamic discovery via callable
- **⚡ Two Interfaces**: [Index-based](usage_index_vs_direct.md#index-interface) (Discrete actions) or [direct](usage_index_vs_direct.md#direct-interface) Option passing  
- **⏱️ Simple Time Semantics**: Each primitive action = 1 tick, [duration = k_exec](durations.md)
- **🎭 Action Masking**: Support for [discrete action availability](masking_and_precheck.md)
- **📊 Rich Info**: Comprehensive execution metadata in `info["smdp"]`
- **🛡️ Error Handling**: Detailed [validation and runtime errors](errors.md)
- **🔄 Continuous Actions**: Full support for continuous action spaces
- **🎲 Built-in Defaults**: Ready-to-use option generators and reward aggregators

## Quick Start

### Index Interface (Recommended for RL)

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option

# Create environment and define options
env = gym.make("CartPole-v1")
options = [
    Option(actions=[0, 0, 1], name="left-left-right"),   # 3 actions = 3 ticks
    Option(actions=[1, 1, 0], name="right-right-left"),  # 3 actions = 3 ticks
    Option(actions=[0, 1], name="left-right"),           # 2 actions = 2 ticks
]

# Wrap with SMDPfier
smdp_env = SMDPfier(
    env,
    options_provider=options,       # Static options list
    action_interface="index",       # Discrete(3) action space
    max_options=len(options)
)

# Use like any Gym environment
obs, info = smdp_env.reset()
obs, reward, term, trunc, info = smdp_env.step(0)  # Execute first option

# Access SMDP metadata
smdp = info["smdp"]
print(f"Option '{smdp['option']['name']}' executed {smdp['k_exec']} steps")
print(f"Duration: {smdp['duration']} ticks (= k_exec)")
print(f"Per-step rewards: {smdp['rewards']}")

# Apply SMDP discounting
gamma = 0.99
discounted_reward = reward * (gamma ** smdp['duration'])
```

### Direct Interface (Intuitive)

```python
# Pass Option objects directly
smdp_env = SMDPfier(env, options_provider=options, action_interface="direct")

# Execute with Option object
obs, reward, term, trunc, info = smdp_env.step(options[0])
```

## Core Concepts

### Options
**Options** are sequences of primitive actions that are executed atomically:

```python
# An option with 3 primitive actions
option = Option(
    actions=[0, 1, 0],           # Action sequence
    name="left-right-left",      # Human-readable name
    meta={"strategy": "zigzag"}  # Optional metadata
)
```

### Time Semantics (v0.2.0+)

**Simple and natural:**
- Each primitive action = **1 tick** of time
- Option duration = **k_exec** (number of primitive actions executed)
- If option completes: `duration = len(option.actions)`
- If terminated early: `duration < len(option.actions)`

**Example:**
```python
option = Option([0, 1, 0], "three-action-option")  # 3 actions

# If it completes normally: duration = 3 ticks
# If episode terminates after 2 actions: duration = 2 ticks
```

### SMDP Discounting

**Standard MDP**: `γ^{1}` per primitive step  
**SMDP**: `γ^{k}` where k = option duration

```python
# MDP: Each primitive step discounts by γ
mdp_return = r₁ + γ¹·r₂ + γ²·r₃ + γ³·r₄

# SMDP: Each option discounts by γ^{duration}
# Options with lengths [3, 2, 4]:  
smdp_return = r₁ + γ³·r₂ + γ⁵·r₃ + γ⁹·r₄
#                   ↑      ↑       ↑
#                   3    3+2    3+2+4
```

### Action Interfaces

SMDPfier provides two ways to select options:

**[Index Interface](usage_index_vs_direct.md#index-interface)** (Recommended for RL)
```python
action_interface="index"  
# Creates Discrete(max_options) action space
# Use integer indices: env.step(0), env.step(1), etc.
```

**[Direct Interface](usage_index_vs_direct.md#direct-interface)** (Intuitive)
```python
action_interface="direct"
# Pass Option objects directly: env.step(option)
```

### SMDP Info Payload

Every step returns comprehensive metadata in `info["smdp"]`:

```python
{
    "option": {
        "id": "abc123...",           # Stable hash-based ID  
        "name": "left-right-left",   # Human-readable name
        "len": 3,                    # Number of actions
        "meta": {}                   # User metadata
    },
    "k_exec": 3,                     # Steps actually executed
    "duration": 3,                   # Duration in ticks (= k_exec)
    "rewards": [1.0, 1.0, 1.0],     # Per-step rewards
    "terminated_early": False,       # Episode ended during option?
    "action_mask": [1, 1, 0],       # Available options (index interface only)
    "num_dropped": 0                 # Dropped options (index interface only)
}
```

See the [API Reference](api.md#smdp-info-payload-structure) for complete details.

## Documentation Guide

| Section | Focus | When to Read |
|---------|-------|--------------|
| **[API Reference](api.md)** | Complete SMDPfier API | Setting up your wrapper |
| **[Durations Guide](durations.md)** | Duration = k_exec, SMDP discounting | Understanding time semantics |
| **[Index vs Direct](usage_index_vs_direct.md)** | Interface comparison | Choosing action interface |
| **[Masking & Precheck](masking_and_precheck.md)** | Action constraints | Handling invalid actions |
| **[Error Handling](errors.md)** | Debugging failed options | Troubleshooting |
| **[FAQ](faq.md)** | Common questions | Quick answers |
| **[Migration from 0.1.x](migration_0_2.md)** | Upgrading to v0.2.0 | Updating existing code |

## Quick Navigation

**🚀 New to SMDPfier?** Start with the [Quick Start](#quick-start) above and [FAQ](faq.md).

**🤖 Building an RL agent?** See [Index Interface](usage_index_vs_direct.md#index-interface) and [Durations](durations.md#smdp-discounting).

**🔧 Need custom behavior?** Check [API Reference](api.md) and [examples/](../examples/).

**❓ Something not working?** Try [Error Handling](errors.md) and [FAQ](faq.md).

**⬆️ Upgrading from 0.1.x?** See [Migration Guide](migration_0_2.md).

---

## Installation

```bash
pip install smdpfier
```

For development:
```bash
git clone https://github.com/smdpfier/smdpfier.git
cd smdpfier
pip install -e .[dev]
```

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Usage Guide](usage_index_vs_direct.md) - Interface comparison
- [Durations](durations.md) - Understanding duration = k_exec
- [FAQ](faq.md) - Common questions and troubleshooting
- [Migration Guide](migration_0_2.md) - Upgrading from 0.1.x
- [Examples](../examples/) - Working code examples
