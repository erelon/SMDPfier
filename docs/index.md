# SMDPfier Documentation

Welcome to SMDPfier, a Gymnasium wrapper that enables **Semi-Markov Decision Process (SMDP)** behavior in reinforcement learning environments through **Options** and **duration metadata**.

## Overview

SMDPfier transforms any Gymnasium environment into an SMDP by allowing agents to execute **Options** (sequences of primitive actions) while tracking **duration metadata** in abstract "ticks" for proper temporal discounting.

**🎯 Key Insight**: SMDPfier separates **execution** (how many `env.step()` calls) from **time** (abstract ticks for discounting), enabling true SMDP learning with γ^{ticks}.

## Key Features

- **🔗 Flexible Options**: Static sequences or dynamic discovery via callable
- **⚡ Two Interfaces**: [Index-based](usage_index_vs_direct.md#index-interface) (Discrete actions) or [direct](usage_index_vs_direct.md#direct-interface) Option passing  
- **⏱️ Duration Metadata**: Integer ticks for [true SMDP discounting](durations.md)
- **🎭 Action Masking**: Support for [discrete action availability](masking_and_precheck.md)
- **📊 Rich Info**: Comprehensive execution metadata in `info["smdp"]`
- **🛡️ Error Handling**: Detailed [validation and runtime errors](errors.md)
- **🔄 Continuous Actions**: Full support for continuous action spaces
- **🎲 Built-in Defaults**: Ready-to-use option generators and duration functions

## Quick Start

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration

# Create environment and define options
env = gym.make("CartPole-v1")
options = [
    Option(actions=[0, 0, 1], name="left-left-right"),   # 3 steps
    Option(actions=[1, 1, 0], name="right-right-left"),  # 3 steps
    Option(actions=[0, 1], name="left-right"),           # 2 steps
]

# Wrap with SMDPfier
smdp_env = SMDPfier(
    env,
    options_provider=options,               # Static options list
    duration_fn=ConstantOptionDuration(10), # 10 ticks per option
    action_interface="index"                # Discrete(3) action space
)

# Use like any Gym environment
obs, info = smdp_env.reset()
obs, reward, term, trunc, info = smdp_env.step(0)  # Execute first option

# Access SMDP metadata
smdp = info["smdp"]
print(f"Option '{smdp['option']['name']}' executed {smdp['k_exec']}/3 steps")
print(f"Duration: {smdp['duration_exec']} ticks (for SMDP discounting)")
print(f"Per-step rewards: {smdp['rewards']}")

# Apply SMDP discounting
gamma = 0.99
discounted_reward = reward * (gamma ** smdp['duration_exec'])
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

### Steps vs Duration (Critical Distinction!)

**This is the most important concept in SMDPfier:**

| Concept | Definition | Controls | Purpose |
|---------|------------|----------|---------|
| **Steps** | Number of `env.step()` calls | Option length | Environment execution |
| **Duration (Ticks)** | Abstract time units | Duration function | SMDP discounting only |

**Example:**
```python
option = Option([0, 1, 0], "three-action-option")  # Always 3 steps
duration_fn = ConstantOptionDuration(10)            # Always 10 ticks

# Result: 3 environment steps executed, 10 ticks of "time" for discounting
```

**Duration does NOT control execution** - it's purely metadata for temporal reasoning.

### SMDP Discounting

**Standard MDP**: `γ^{steps}`  
**SMDP**: `γ^{ticks}`

```python
# MDP: Each step advances time by 1
mdp_return = r₁ + γ¹·r₂ + γ²·r₃ + γ³·r₄

# SMDP: Each option advances time by its duration
# Options with durations [5, 3, 7] ticks:  
smdp_return = r₁ + γ⁵·r₂ + γ⁸·r₃ + γ¹⁵·r₄
#                   ↑      ↑       ↑
#                   5    5+3    5+3+7
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
    "rewards": [1.0, 1.0, 1.0],     # Per-step rewards
    "duration_planned": 10,          # Expected ticks
    "duration_exec": 10,             # Actual ticks (may differ due to early termination)
    "terminated_early": False,       # Episode ended during option?
    "time_units": "ticks",           # Always "ticks"
}
```

See the [API Reference](api.md#smdp-info-payload-structure) for complete details.

## Documentation Guide

| Section | Focus | When to Read |
|---------|-------|--------------|
| **[API Reference](api.md)** | Complete SMDPfier API | Setting up your wrapper |
| **[Durations Guide](durations.md)** | Ticks, SMDP discounting, policies | Understanding temporal mechanics |
| **[Index vs Direct](usage_index_vs_direct.md)** | Interface comparison | Choosing action interface |
| **[Masking & Precheck](masking_and_precheck.md)** | Action constraints | Handling invalid actions |
| **[Error Handling](errors.md)** | Debugging failed options | Troubleshooting |
| **[FAQ](faq.md)** | Common questions | Quick answers |

## Quick Navigation

**🚀 New to SMDPfier?** Start with the [Quick Start](#quick-start) above and [FAQ](faq.md).

**🤖 Building an RL agent?** See [Index Interface](usage_index_vs_direct.md#index-interface) and [Durations](durations.md#smdp-discounting).

**🔧 Need custom behavior?** Check [API Reference](api.md) and [examples/](../examples/).

**❓ Something not working?** Try [Error Handling](errors.md) and [FAQ](faq.md).

---

**Next:** [API Reference](api.md) | **Examples:** [../examples/](../examples/)

### Options
Options represent temporal abstractions - sequences of primitive actions executed atomically. Each option has:
- `actions`: Sequence of primitive actions
- `name`: Human-readable identifier  
- `meta`: Optional metadata dictionary

### Durations
Durations are metadata (in ticks) that don't control execution but enable true SMDP discounting:
- **Planned**: Expected duration before execution
- **Executed**: Actual duration after execution (may differ due to early termination)

### Interfaces
- **Index Interface**: Choose options by index from `Discrete(max_options)` action space
- **Direct Interface**: Pass `Option` objects directly to `step()`

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

## Examples

### CartPole with Static Options (Index Interface)

```python
import gymnasium as gym
from smdpfier import Option, SMDPfier
from smdpfier.defaults import ConstantOptionDuration, sum_rewards

# Create base environment
env = gym.make("CartPole-v1")

# Define static options
static_options = [
    Option([0, 0, 1], "left-left-right", meta={"category": "mixed"}),
    Option([1, 1, 0], "right-right-left", meta={"category": "mixed"}), 
    Option([0, 0, 0], "left-triple", meta={"category": "directional"}),
    Option([1, 1, 1], "right-triple", meta={"category": "directional"}),
]

# Create SMDPfier with index interface
smdp_env = SMDPfier(
    env,
    options_provider=static_options,
    duration_fn=ConstantOptionDuration(10),  # 10 ticks per option
    reward_agg=sum_rewards,
    action_interface="index",
    max_options=len(static_options),
)

# Execute
obs, info = smdp_env.reset(seed=42)
obs, reward, terminated, truncated, info = smdp_env.step(0)

# Check results
smdp_info = info["smdp"]
print(f"Executed option: {smdp_info['option']['name']}")
print(f"Steps: {smdp_info['k_exec']}/{smdp_info['option']['len']}")
print(f"Duration: {smdp_info['duration_exec']} ticks")
```

### Taxi with Dynamic Options & Masking (Index Interface)

```python
import gymnasium as gym  
from smdpfier import Option, SMDPfier
from smdpfier.defaults import RandomActionDuration, mean_rewards

def create_taxi_options(obs, info):
    """Dynamic option provider based on current state."""
    return [
        Option([0], "south", meta={"type": "primitive"}),
        Option([1], "north", meta={"type": "primitive"}),
        Option([2], "east", meta={"type": "primitive"}),
        Option([3], "west", meta={"type": "primitive"}),
        Option([4], "pickup", meta={"type": "primitive"}),
        Option([5], "dropoff", meta={"type": "primitive"}),
        # Navigation sequences
        Option([0, 2], "south-east", meta={"type": "navigation"}),
        Option([1, 3], "north-west", meta={"type": "navigation"}),
    ]

def taxi_availability_function(obs):
    """Restrict certain actions based on state."""
    # Movement always available
    available = [0, 1, 2, 3]
    # Conditionally add pickup/dropoff
    if (obs + 42) % 10 < 7:  # Pseudo-random condition
        available.append(4)  # pickup
    if (obs + 17) % 10 < 6:
        available.append(5)  # dropoff
    return available

# Create SMDPfier with masking
env = gym.make("Taxi-v3")
smdp_env = SMDPfier(
    env,
    options_provider=create_taxi_options,
    duration_fn=RandomActionDuration(3, 8),
    reward_agg=mean_rewards,
    action_interface="index",
    max_options=12,
    availability_fn=taxi_availability_function,
    precheck=True,
)

obs, info = smdp_env.reset(seed=42)

# Check masking
mask = info["action_mask"]
available_options = [i for i, avail in enumerate(mask) if avail]
print(f"Available options: {available_options}")

obs, reward, terminated, truncated, info = smdp_env.step(available_options[0])
print(f"Mean reward: {reward:.3f}")
```

### Pendulum with Continuous Actions (Direct Interface)

```python
import gymnasium as gym
from smdpfier import Option, SMDPfier  
from smdpfier.defaults import ConstantActionDuration, discounted_sum

# Create continuous action options
continuous_options = [
    Option([[1.0], [-1.0], [1.0], [-1.0]], "oscillate-high", 
           meta={"category": "oscillation"}),
    Option([[0.5], [-0.5], [0.5]], "oscillate-medium",
           meta={"category": "oscillation"}), 
    Option([[0.0], [0.0], [0.0]], "hold-steady",
           meta={"category": "stabilization"}),
]

# Create SMDPfier with direct interface  
env = gym.make("Pendulum-v1")
smdp_env = SMDPfier(
    env,
    options_provider=continuous_options,
    duration_fn=ConstantActionDuration(4),  # 4 ticks per action
    reward_agg=discounted_sum,
    action_interface="direct",
)

obs, info = smdp_env.reset(seed=42)

# Execute by passing Option objects directly
option_to_execute = continuous_options[0]  # oscillate-high
obs, reward, terminated, truncated, info = smdp_env.step(option_to_execute)

smdp_info = info["smdp"]
print(f"Executed {smdp_info['k_exec']} actions")
print(f"Total duration: {smdp_info['duration_exec']} ticks")
print(f"Discounted reward: {reward:.3f}")
```

## Next Steps

- [API Reference](api.md) - Complete API documentation
- [Usage Guide](usage_index_vs_direct.md) - Interface comparison and examples
- [Durations](durations.md) - Understanding duration metadata and SMDP discounting
- [FAQ](faq.md) - Common questions and troubleshooting
