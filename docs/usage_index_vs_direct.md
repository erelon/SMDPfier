# Index vs Direct Interfaces

SMDPfier provides two interfaces for option selection, each optimized for different use cases. Understanding when to use each interface is crucial for effective SMDP implementation.

## Interface Comparison

| Aspect | Index Interface | Direct Interface |
|--------|----------------|------------------|
| **Action Space** | `Discrete(max_options)` | Same as Option actions |
| **Actions** | Integer indices (0, 1, 2, ...) | Option objects |
| **Best For** | Reinforcement Learning | Scripting/Testing |
| **Action Masking** | Built-in support | Not applicable |
| **Dynamic Options** | Supported with overflow handling | Straightforward |
| **RL Integration** | Seamless | Requires adaptation |
| **Debugging** | Index-based (less intuitive) | Object-based (more intuitive) |

## Index Interface

The index interface transforms SMDPfier into a **discrete action space** where each action is an integer index selecting an available option.

### When to Use Index Interface

✅ **Choose Index Interface When:**
- Training RL agents (most algorithms expect discrete actions)
- Need action masking based on environment state
- Working with dynamic option sets
- Integrating with existing RL frameworks (Stable-Baselines3, RLLib, etc.)
- Want built-in overflow handling for dynamic options

### Basic Index Interface Setup

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration

# Define static options
options = [
    Option([0, 0, 1], "left-left-right"),
    Option([1, 1, 0], "right-right-left"),
    Option([0, 1, 0], "left-right-left"),
]

# Create SMDPfier with index interface
env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="index",  # Default
    max_options=len(options)   # Must specify for static options
)

print(f"Original action space: Discrete(2)")
print(f"SMDP action space: {env.action_space}")  # Discrete(3)

# Use integer actions
obs, info = env.reset()
action = 1  # Select second option ("right-right-left")
obs, reward, term, trunc, info = env.step(action)

print(f"Executed option: {info['smdp']['option']['name']}")
```

### Action Masking with Index Interface

Action masking allows you to restrict which options are available based on the current environment state.

```python
def cart_availability(obs):
    """Restrict options based on cart position."""
    cart_position = obs[0]
    
    if cart_position > 0.5:
        return [0, 2]  # Only left-based options when cart is far right
    elif cart_position < -0.5:
        return [1]     # Only right-based options when cart is far left
    else:
        return [0, 1, 2]  # All options available in center

env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="index",
    max_options=3,
    availability_fn=cart_availability
)

obs, info = env.reset()
print(f"Available actions: {info['smdp']['action_mask']}")
# Might show: [1, 1, 1] (all available) or [1, 0, 1] (middle option masked)
```

### Dynamic Options with Index Interface

Dynamic options change based on the current state, requiring careful overflow handling.

```python
from smdpfier.defaults.options import RandomStaticLen

def dynamic_options_generator(obs, info):
    """Generate different options based on cart velocity."""
    velocity = obs[1]
    
    if abs(velocity) > 0.1:  # Fast movement
        # Short options for quick corrections
        return [
            Option([0], "quick-left"),
            Option([1], "quick-right"),
        ]
    else:  # Slow movement  
        # Longer options for building momentum
        return [
            Option([0, 0, 0], "triple-left"),
            Option([1, 1, 1], "triple-right"),
            Option([0, 1, 0], "left-right-left"),
            Option([1, 0, 1], "right-left-right"),
        ]

env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=dynamic_options_generator,
    duration_fn=ConstantOptionDuration(3),
    action_interface="index",
    max_options=4  # Maximum expected options
)

obs, info = env.reset()
# If generator returns 2 options but max_options=4:
# - Actions 0,1 are valid
# - Actions 2,3 are masked out
# - info["smdp"]["action_mask"] = [1, 1, 0, 0]
```

### Overflow Handling

When dynamic options exceed `max_options`, SMDPfier applies **truncate policy** by default:

```python
# Generator returns 6 options, but max_options=4
available_options = dynamic_generator(obs, info)  # Returns 6 options
# Result: First 4 options are used, 2 are dropped
# info["smdp"]["num_dropped"] = 2
```

## Direct Interface

The direct interface allows you to pass `Option` objects directly to `env.step()`, providing an intuitive and flexible approach.

### When to Use Direct Interface

✅ **Choose Direct Interface When:**
- Scripting or testing specific option sequences
- Building non-RL controllers or heuristics
- Debugging option behavior
- Prototyping before RL training
- Need full control over option selection

### Basic Direct Interface Setup

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantActionDuration

# Define options
options = [
    Option([0, 0, 1], "left-left-right"),
    Option([1, 1, 0], "right-right-left"),
    Option([0, 1, 0, 1], "alternating"),
]

# Create SMDPfier with direct interface
env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=options,
    duration_fn=ConstantActionDuration(2),  # 2 ticks per action
    action_interface="direct"
)

print(f"Action space: {env.action_space}")  # Same as original env

# Use Option objects directly
obs, info = env.reset()
option = options[1]  # Select "right-right-left"
obs, reward, term, trunc, info = env.step(option)

print(f"Executed {info['smdp']['k_exec']} steps")
print(f"Duration: {info['smdp']['duration_exec']} ticks")
```

### Dynamic Options with Direct Interface

```python
def get_option_for_state(obs):
    """Select option based on current state."""
    cart_position, cart_velocity = obs[0], obs[1]
    
    if cart_position > 0 and cart_velocity > 0:
        return Option([0, 0], "strong-left")  # Moving right, correct strongly
    elif cart_position < 0 and cart_velocity < 0:
        return Option([1, 1], "strong-right")  # Moving left, correct strongly
    else:
        return Option([0, 1], "gentle-correction")  # Gentle correction

# Simple control loop
obs, info = env.reset()
for step in range(100):
    option = get_option_for_state(obs)
    obs, reward, term, trunc, info = env.step(option)
    
    if term or trunc:
        break
```

### Continuous Actions with Direct Interface

The direct interface works seamlessly with continuous action spaces:

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration

# Continuous action options
continuous_options = [
    Option([[-1.0], [0.0], [1.0]], "left-center-right"),
    Option([[0.5], [0.5]], "gentle-right"),
    Option([[-2.0]], "hard-left"),
]

env = SMDPfier(
    gym.make("Pendulum-v1"),
    options_provider=continuous_options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="direct"
)

obs, info = env.reset()
option = continuous_options[0]
obs, reward, term, trunc, info = env.step(option)
```

## Interface Selection Guide

### Choose Index Interface When:

```python
# RL Training
env = SMDPfier(..., action_interface="index", max_options=10)
agent.learn(env)  # Works with any RL algorithm

# Action Masking Needed
env = SMDPfier(..., action_interface="index", availability_fn=mask_fn)

# Dynamic Options with Overflow
env = SMDPfier(..., action_interface="index", max_options=20)
# Handles varying option counts gracefully
```

### Choose Direct Interface When:

```python
# Scripted Control
for situation in test_cases:
    option = select_option_for_situation(situation)
    obs, reward, term, trunc, info = env.step(option)

# Debugging Specific Options
problem_option = Option([0, 1, 0], "problematic-sequence")
obs, reward, term, trunc, info = env.step(problem_option)

# Prototyping Before RL
def human_policy(obs):
    return Option([best_action_for(obs)], "human-choice")
```

## Configuration Examples

### Index Interface Configuration

```python
# Static options with masking
env = SMDPfier(
    base_env,
    options_provider=static_options,
    duration_fn=ConstantOptionDuration(10),
    action_interface="index",
    max_options=len(static_options),
    availability_fn=masking_function,
    precheck=True  # Validate options before execution
)

# Dynamic options with overflow handling
env = SMDPfier(
    base_env,
    options_provider=dynamic_generator,
    duration_fn=RandomActionDuration(2, 5),
    action_interface="index", 
    max_options=15,  # Allow up to 15 options
    # Overflow: truncate to first 15, record num_dropped
)
```

### Direct Interface Configuration

```python
# Simple direct interface
env = SMDPfier(
    base_env,
    options_provider=option_list,
    duration_fn=ConstantActionDuration(3),
    action_interface="direct"
    # No max_options needed
    # No availability_fn needed
)
```

## Summary

| Use Case | Recommended Interface | Why |
|----------|----------------------|-----|
| **RL Training** | Index | Discrete action space expected |
| **Action Masking** | Index | Built-in masking support |
| **Dynamic Options** | Index | Overflow handling |
| **Scripting/Testing** | Direct | Intuitive option passing |
| **Debugging** | Direct | Clear option identification |
| **Continuous Actions** | Direct | Natural continuous support |
| **Prototyping** | Direct | Flexible experimentation |

**Most Common Pattern:**
1. Start with **direct interface** for prototyping and testing
2. Switch to **index interface** when training RL agents

---

**Next**: [Masking and Precheck](masking_and_precheck.md) | **See Also**: [API Reference](api.md#action-interfaces)
