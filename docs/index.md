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

## Option Types

SMDPfier supports two types of options:

### ListOption (Fixed Sequences)

Execute a predetermined sequence of actions:

```python
from smdpfier import Option  # Factory function creates ListOption

# Basic usage
option = Option([0, 1, 0], "left-right-left")

# With metadata
option = Option(
    [0, 0, 1, 1],
    "double-pairs",
    meta={"category": "symmetric"}
)

# Continuous actions
option = Option(
    [[-1.0], [0.5], [2.0]],
    "continuous-sequence"
)
```

### Stateful Options (Adaptive Behavior)

Create options that observe the environment and adapt their behavior:

```python
from smdpfier.option import OptionBase

class ThresholdOption(OptionBase):
    """Execute action until observation exceeds threshold."""
    
    def __init__(self, threshold=0.5, max_steps=5):
        self.threshold = threshold
        self.max_steps = max_steps
        self.step_count = 0
    
    def begin(self, obs, info):
        """Initialize state."""
        self.step_count = 0
    
    def act(self, obs, info):
        """Choose action based on current observation."""
        # Terminate if threshold exceeded
        if obs[0] > self.threshold:
            return None  # Terminate without action
        
        # Continue with action 0
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        return 0, done
    
    def on_step(self, obs, reward, terminated, truncated, info):
        """Process step result."""
        pass  # Could track statistics here
    
    def preview(self, obs, info):
        """Preview first action for masking."""
        return None if obs[0] > self.threshold else 0
    
    def identity(self):
        """Stable identity for hashing."""
        return ("ThresholdOption", str(self.threshold), str(self.max_steps))
    
    @property
    def name(self):
        return f"threshold_{self.threshold}"

# Usage
env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=[ThresholdOption(threshold=0.3)],
    action_interface="index"
)
```

**Key Benefits of Stateful Options:**
- **Observe environment**: Access current observation in `act()`
- **Adaptive duration**: Terminate early via `done=True` or return `None`
- **Immediate termination**: Return `None` to skip action execution (`duration=0`)
- **State tracking**: Use `on_step()` to collect data or update internal state

See [API Reference](api.md#custom-stateful-options) for complete stateful option documentation.

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
