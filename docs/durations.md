# Durations and SMDP Discounting

**Duration** in SMDPfier v0.2.0+ is simple: each primitive action = 1 tick, and option duration equals the number of primitive actions executed (k_exec).

## 🎯 Time Semantics (v0.2.0+)

**The key principle:** Each primitive action = 1 tick of time.

| Concept | Value |
|---------|-------|
| **Tick** | One primitive action execution |
| **Option Duration** | Number of primitive actions executed (k_exec) |
| **Complete Execution** | duration = `len(option)` for fixed-length options |
| **Partial Execution** | duration = number of actions before termination |
| **Stateful Termination** | Option can stop early via `done=True` from `act()` |

### Simple Example

```python
from smdpfier import SMDPfier, Option
import gymnasium as gym

env = gym.make("CartPole-v1")
options = [
    Option([0, 1, 0], "three-step"),      # 3 actions = 3 ticks
    Option([1, 1, 0, 0], "four-step"),    # 4 actions = 4 ticks
    Option([0, 1], "two-step"),           # 2 actions = 2 ticks
]

smdp_env = SMDPfier(env, options_provider=options, action_interface="index", max_options=3)

# Execute first option (3 actions)
obs, reward, term, trunc, info = smdp_env.step(0)

# Check duration
assert info["smdp"]["duration"] == 3  # 3 actions executed = 3 ticks
assert info["smdp"]["k_exec"] == 3    # Same value
```

### Visual Example

```
Fixed Option: [0, 1, 0, 1, 1]  (5 actions)

Normal Execution:
┌─────┬─────┬─────┬─────┬─────┐
│ a=0 │ a=1 │ a=0 │ a=1 │ a=1 │
└─────┴─────┴─────┴─────┴─────┘
tick:   1     2     3     4     5    → duration = 5

Episode Termination (after 3 actions):
┌─────┬─────┬─────┐  X
│ a=0 │ a=1 │ a=0 │ episode ends
└─────┴─────┴─────┘
tick:   1     2     3                → duration = 3

Stateful Option Early Stop (done=True):
┌─────┬─────┐  option signals done=True
│ a=0 │ a=1 │
└─────┴─────┘
tick:   1     2                      → duration = 2
```

## 🎮 Stateful Option Duration Control

Stateful options can control their own duration by returning `done=True`:

```python
from smdpfier.option import OptionBase

class AdaptiveOption(OptionBase):
    """Option that stops when observation exceeds threshold."""
    
    def __init__(self, threshold=0.5, max_steps=5):
        self.threshold = threshold
        self.max_steps = max_steps
        self.step_count = 0
    
    def begin(self, obs, info):
        self.step_count = 0
    
    def act(self, obs, info):
        if obs[0] > self.threshold:
            return None  # Terminate immediately (duration=0)
        
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        return 0, done  # Signal done when max_steps reached
    
    def on_step(self, obs, reward, terminated, truncated, info):
        pass
    
    # ... other methods ...

# Usage
env = SMDPfier(base_env, options_provider=[AdaptiveOption()])
obs, reward, term, trunc, info = env.step(0)

# Duration varies based on when option decided to stop
print(f"Duration: {info['smdp']['duration']}")  # Could be 0, 1, 2, ..., or max_steps
```

### Duration Scenarios for Stateful Options

1. **Immediate Termination** (action is `None`):
   ```python
   def act(self, obs, info):
       return None  # duration = 0, k_exec = 0
   ```

2. **Early Stop** (done=True):
   ```python
   def act(self, obs, info):
       if self.should_stop(obs):
           return action, True  # duration = k_exec when stopped
       return action, False
   ```

3. **Episode Ends**:
   - Option stops when environment returns `terminated=True` or `truncated=True`
   - Duration = k_exec at termination point

## 🔄 SMDP Discounting

With simple duration = k_exec, SMDP discounting becomes straightforward:

### Basic Discounting

```python
gamma = 0.99

# Execute option
obs, reward, term, trunc, info = env.step(action)
duration = info["smdp"]["duration"]

# Apply SMDP discount
discounted_reward = reward * (gamma ** duration)
```

### Multi-Step Example

```python
# Execute sequence of options
gamma = 0.99
total_time = 0
cumulative_return = 0

for action in [0, 1, 2]:
    obs, reward, term, trunc, info = env.step(action)
    duration = info["smdp"]["duration"]
    
    # Apply time-cumulative discount
    discount_factor = gamma ** total_time
    cumulative_return += reward * discount_factor
    
    total_time += duration
    if term or trunc:
        break

print(f"Total time: {total_time} ticks")
print(f"Cumulative return: {cumulative_return}")
```

### Comparison: MDP vs SMDP

```python
# MDP: Each primitive action discounts by γ
mdp_return = r₀ + γ¹·r₁ + γ²·r₂ + γ³·r₃ + γ⁴·r₄

# SMDP: Each option discounts by γ^{duration}
# Options with lengths [3, 2, 4]:
smdp_return = R₀ + γ³·R₁ + γ⁵·R₂
#                   ↑      ↑
#                   3    3+2
```

## 📊 Reward Aggregation

By default, SMDPfier sums the per-step rewards to produce the macro reward:

```python
# Default behavior (v0.2.0+)
macro_reward = sum(primitive_rewards)

# Example:
# Option executes 3 steps with rewards [1.0, 1.0, 1.0]
# macro_reward = 1.0 + 1.0 + 1.0 = 3.0
```

### Custom Aggregators

You can customize reward aggregation:

```python
from smdpfier import SMDPfier
from smdpfier.defaults import mean_rewards, discounted_sum

# Average per-step rewards
env = SMDPfier(env, options_provider=options, reward_agg=mean_rewards)
# 3 steps with rewards [1.0, 1.0, 1.0] → macro_reward = 1.0

# Discount per-step rewards
env = SMDPfier(env, options_provider=options, reward_agg=discounted_sum(gamma=0.99))
# 3 steps with rewards [1.0, 1.0, 1.0] → macro_reward = 1.0 + 0.99 + 0.98 = 2.97

# Custom aggregator
def custom_agg(rewards):
    return sum(r * (i + 1) for i, r in enumerate(rewards))  # Weighted sum

env = SMDPfier(env, options_provider=options, reward_agg=custom_agg)
```

## ⚠️ Early Termination

When the episode terminates before an option completes, duration reflects actual execution:

```python
option = Option([0, 1, 0, 1, 1], "five-step")

# Episode terminates after 3rd action
# - k_exec = 3
# - duration = 3 (not 5!)
# - rewards = [r₁, r₂, r₃] (only 3 rewards)
# - terminated_early = True

obs, reward, term, trunc, info = env.step(option)
if info["smdp"]["terminated_early"]:
    print(f"Option interrupted after {info['smdp']['k_exec']} of {info['smdp']['option']['len']} actions")
    print(f"Duration: {info['smdp']['duration']} ticks (not {info['smdp']['option']['len']})")
```

### Why This Matters for Discounting

```python
# Option length: 5 actions
# Terminated after 3 actions

# ✅ Correct: Use actual duration
duration = info["smdp"]["duration"]  # = 3
next_discount = gamma ** duration    # = γ³

# ❌ Wrong: Use planned duration
planned = info["smdp"]["option"]["len"]  # = 5
wrong_discount = gamma ** planned         # = γ⁵ (incorrect!)
```

## 🧪 Full Example

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option

# Setup
env = gym.make("CartPole-v1")
options = [
    Option([0, 0, 1], "left-left-right"),    # 3 ticks
    Option([1, 1, 0, 0], "right-right-left-left"),  # 4 ticks
]

smdp_env = SMDPfier(
    env,
    options_provider=options,
    action_interface="index",
    max_options=2
)

# Episode
obs, info = smdp_env.reset()
gamma = 0.99
total_time = 0
returns = []

for step in range(10):
    action = smdp_env.action_space.sample()
    obs, reward, term, trunc, info = smdp_env.step(action)
    
    smdp = info["smdp"]
    duration = smdp["duration"]
    
    print(f"Step {step}:")
    print(f"  Option: {smdp['option']['name']}")
    print(f"  Duration: {duration} ticks")
    print(f"  Rewards: {smdp['rewards']}")
    print(f"  Macro reward: {reward}")
    print(f"  Discounted: {reward * (gamma ** total_time):.4f}")
    
    total_time += duration
    returns.append(reward * (gamma ** total_time))
    
    if term or trunc:
        break

print(f"\nTotal time: {total_time} ticks")
print(f"Total return: {sum(returns):.4f}")
```

## 📝 Key Takeaways

1. **Each primitive action = 1 tick**: Time is simple and natural
2. **Duration = k_exec**: No separate duration function needed
3. **Early termination**: Duration reflects actual execution, not planned
4. **SMDP discounting**: Use `γ^{duration}` for each option
5. **Macro reward**: Sum of per-step rewards by default (customizable)

## 🔄 Migrating from 0.1.x

In v0.1.x, you had to specify a `duration_fn`. In v0.2.0+, this is automatic:

```python
# 0.1.x - Old way
from smdpfier.defaults import ConstantActionDuration

env = SMDPfier(
    env,
    options_provider=options,
    duration_fn=ConstantActionDuration(1),  # Each action = 1 tick
    # ...
)

# 0.2.0+ - New way (automatic!)
env = SMDPfier(
    env,
    options_provider=options,
    # duration_fn removed - each action automatically = 1 tick
    # ...
)
```

See [Migration Guide](migration_0_2.md) for complete upgrade instructions.

---

**Next:** [FAQ](faq.md) | **Previous:** [API Reference](api.md)
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
