# Durations and SMDP Discounting

**Duration metadata** is the key feature that enables true Semi-Markov Decision Process (SMDP) learning with proper temporal discounting using γ^{ticks}.

## 🎯 The Critical Distinction: Steps vs Duration (Ticks)

**This is the most important concept in SMDPfier.** Understanding this distinction is essential for proper SMDP learning.

| Aspect | Steps | Duration (Ticks) |
|--------|-------|------------------|
| **Definition** | Number of `env.step()` calls | Abstract time units |
| **Determined By** | Option length (`len(option.actions)`) | Duration function |
| **Controls** | Environment execution | Nothing (metadata only) |
| **Used For** | Running the environment | SMDP discounting (γ^{ticks}) |
| **Always Equal To** | Number of actions in option | Whatever duration function returns |

### Visual Example

```python
option = Option([0, 1, 0], "left-right-left")  # 3 actions
duration_fn = ConstantOptionDuration(10)        # 10 ticks

# Execution:
# Step 1: env.step(0) → reward₁  }
# Step 2: env.step(1) → reward₂  } 3 environment steps
# Step 3: env.step(0) → reward₃  }
# 
# Time accounting: 10 ticks elapsed (for discounting)
```

**Key Point**: The option will ALWAYS execute 3 steps regardless of the duration function. Duration is purely temporal metadata.

## Duration Types

SMDPfier supports two duration formats with different use cases:

### Scalar Duration (Option-Level)

Returns a single integer representing the total duration for the entire option.

```python
from smdpfier.defaults import ConstantOptionDuration, RandomOptionDuration

# Fixed duration per option
duration_fn = ConstantOptionDuration(10)
# Every option takes 10 ticks, regardless of length

# Random duration per option  
duration_fn = RandomOptionDuration(min_duration=5, max_duration=15)
# Each option gets a random duration between 5-15 ticks
```

**Use Cases:**
- Modeling options as "macro-actions" with fixed time cost
- Simple temporal abstractions
- When option completion time is independent of constituent actions

### List Duration (Action-Level)

Returns a list of integers, one for each primitive action in the option.

```python
from smdpfier.defaults import ConstantActionDuration, RandomActionDuration

# Fixed duration per action
duration_fn = ConstantActionDuration(3)
# Each primitive action takes 3 ticks
# Option [0,1,0] → durations [3,3,3] → total 9 ticks

# Random duration per action
duration_fn = RandomActionDuration(min_duration=2, max_duration=5)  
# Each action gets random duration 2-5 ticks
# Option [0,1,0] → durations [4,2,5] → total 11 ticks
```

**Use Cases:**
- Modeling heterogeneous action costs (e.g., move vs attack)
- Fine-grained temporal modeling
- When different actions have different inherent durations

### Duration Type Comparison

| Scenario | Scalar Duration | List Duration |
|----------|----------------|---------------|
| **Option**: `[0,1,0]` | `10` → 10 ticks total | `[3,4,3]` → 10 ticks total |
| **Early Termination** | Apply partial duration policy | Sum executed portion |
| **Complexity** | Simple | More detailed |
| **Use Case** | Macro-actions | Action-specific costs |

## SMDP Discounting

The power of SMDPs lies in proper temporal discounting using elapsed time rather than step counts.

### Standard MDP vs SMDP

**Standard MDP Discounting:**
```python
# Each step advances time by exactly 1 unit
return = r₁ + γ¹·r₂ + γ²·r₃ + γ³·r₄ + ...
#           ↑    ↑    ↑    ↑
#          t=1  t=2  t=3  t=4
```

**SMDP Discounting:**
```python
# Each option advances time by its duration
# Options with durations [5, 3, 7, 2] ticks:
return = r₁ + γ⁵·r₂ + γ⁸·r₃ + γ¹⁵·r₄ + γ¹⁷·r₅
#           ↑     ↑     ↑      ↑      ↑
#          t=0   t=5   t=8   t=15   t=17
```

### Practical SMDP Discounting Implementation

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration

# Setup
env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=[
        Option([0, 0], "left-left"),     # 2 steps, 10 ticks
        Option([1, 1, 1], "right-x3"),   # 3 steps, 10 ticks  
        Option([0, 1], "left-right"),    # 2 steps, 10 ticks
    ],
    duration_fn=ConstantOptionDuration(10)
)

# Episode with SMDP discounting
gamma = 0.99
discounted_return = 0
elapsed_time = 0

obs, info = env.reset()
for step in range(3):
    action = step % 3  # Cycle through options
    obs, reward, term, trunc, info = env.step(action)
    
    # Get duration from SMDP info
    duration = info["smdp"]["duration_exec"]
    
    # Apply SMDP discounting
    discount_factor = gamma ** elapsed_time
    discounted_return += discount_factor * reward
    
    # Update elapsed time for next option
    elapsed_time += duration
    
    print(f"Option {action}: reward={reward}, duration={duration}, "
          f"discount=γ^{elapsed_time-duration}={discount_factor:.4f}")
    
    if term or trunc:
        break

print(f"Total SMDP discounted return: {discounted_return:.4f}")
```

**Sample Output:**
```
Option 0: reward=2.0, duration=10, discount=γ^0=1.0000
Option 1: reward=3.0, duration=10, discount=γ^10=0.9044  
Option 2: reward=1.0, duration=10, discount=γ^20=0.8179
Total SMDP discounted return: 4.6223
```

## Partial Duration Policies

When an episode terminates early (before an option completes), SMDPfier applies a **partial duration policy** to determine the executed duration.

### Policy Comparison

Consider an option with 3 actions and scalar duration 10, terminating after 2 actions:

| Policy | Formula | Result | Use Case |
|--------|---------|--------|----------|
| `"proportional"` | `(k_exec / option_len) * planned_duration` | `(2/3) * 10 = 6` | **Default** - proportional time |
| `"full"` | `planned_duration` | `10` | Option "completes" conceptually |
| `"zero"` | `0` | `0` | No time passes on failure |

### Example: Partial Duration Calculation

```python
from smdpfier.defaults import ConstantOptionDuration

# Option with 4 actions, 12 ticks planned
option = Option([0, 1, 0, 1], "four-action-option")
duration_fn = ConstantOptionDuration(12)

# Episode terminates after 3 actions (k_exec = 3)

# Proportional policy (default):
duration_exec = (3 / 4) * 12 = 9 ticks

# Full policy:  
duration_exec = 12 ticks

# Zero policy:
duration_exec = 0 ticks
```

### Policy Selection Guide

| Choose | When |
|--------|------|
| `"proportional"` | Time should scale with execution (most realistic) |
| `"full"` | Options have setup costs regardless of completion |  
| `"zero"` | Failed options should not consume time |

## Advanced Duration Functions

### Custom Duration Functions

```python
def custom_duration_fn(option, obs, info):
    """Duration based on option length and environment state."""
    base_duration = len(option.actions) * 2
    
    # Adjust based on environment state  
    if obs[0] > 0:  # Cart position in CartPole
        return base_duration + 5  # Takes longer when cart is right
    else:
        return base_duration

# Usage
env = SMDPfier(env, options_provider=options, duration_fn=custom_duration_fn)
```

### State-Dependent Durations

```python
from smdpfier.defaults import MapActionDuration

# Different actions have different costs
action_duration_map = {
    0: 2,  # Left action: 2 ticks
    1: 5,  # Right action: 5 ticks (more expensive)
}

duration_fn = MapActionDuration(action_duration_map, default_duration=3)

# Option [0, 1, 0] → durations [2, 5, 2] → total 9 ticks
```

## Summary

| Concept | Key Point |
|---------|-----------|
| **Steps vs Ticks** | Steps = execution, Ticks = time metadata |
| **Duration Types** | Scalar (per-option) vs List (per-action) |
| **SMDP Discounting** | Use γ^{duration_exec}, not γ^{steps} |
| **Partial Policies** | Handle early termination gracefully |
| **Custom Functions** | Tailor durations to your domain |

**Remember**: Duration is metadata for discounting - it never affects how many environment steps are executed!

---

**Next**: [Index vs Direct Interfaces](usage_index_vs_direct.md) | **See Also**: [API Reference](api.md#duration-functions)
