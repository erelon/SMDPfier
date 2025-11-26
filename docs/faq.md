# Frequently Asked Questions

Common questions and answers about SMDPfier concepts, usage, and troubleshooting.

## Core Concepts

### What is SMDPfier?

SMDPfier is a Gymnasium wrapper that transforms any environment into a **Semi-Markov Decision Process (SMDP)** by enabling:
- **Options**: Sequences of primitive actions executed atomically
- **Simple Time Semantics**: Each primitive action = 1 tick (v0.2.0+)
- **SMDP Discounting**: Using γ^{k} where k = number of actions executed

**Key Insight (v0.2.0+)**: Each primitive action = 1 tick. Duration = k_exec. Simple and natural.

### How is this different from other hierarchical RL libraries?

| Aspect | SMDPfier | Other Libraries |
|--------|----------|-----------------|
| **Focus** | Option-based SMDP behavior | General hierarchical RL |
| **Complexity** | Single wrapper class | Full frameworks |
| **Environment Support** | Any Gymnasium env unchanged | Often require specific environments |
| **Temporal Modeling** | Each action = 1 tick | Usually step-based |
| **Integration** | Drop-in wrapper | Framework-specific |

### When should I use SMDPfier?

✅ **Use SMDPfier when you want to:**
- Apply **SMDP discounting** with γ^{duration} where duration = actions executed
- Test **hierarchical policies** with temporal abstractions
- Add **option-level control** to existing environments
- **Research temporal abstractions** with simple time semantics

❌ **Don't use SMDPfier when:**
- You need complex option discovery algorithms
- You want full hierarchical RL frameworks (use HRL libraries)
- You don't care about temporal discounting (standard MDP is fine)

## Time Semantics (v0.2.0+)

### How does time work in SMDPfier?

**Simple rule:** Each primitive action = 1 tick.

```python
option = Option([0, 1, 0], "three-actions")  # 3 actions

# Executes 3 primitive actions → duration = 3 ticks
# If terminated early after 2 actions → duration = 2 ticks
```

### Why did duration modeling change in v0.2.0?

**Simplicity and clarity.** The old system separated "steps" from "ticks" which was confusing. Now:
- Each action = 1 tick (natural and intuitive)
- Duration = k_exec (number of actions executed)
- No complex duration functions needed

See [Migration Guide](migration_0_2.md) for upgrading from 0.1.x.

### How do I control option duration?

**Use option length:**

```python
# Want 2 ticks? Use 2 actions:
short_option = Option([0, 1], "short")           # 2 ticks

# Want 5 ticks? Use 5 actions:
long_option = Option([0, 1, 0, 1, 0], "long")    # 5 ticks

# Want 1 tick? Use 1 action:
instant_option = Option([0], "instant")          # 1 tick
```

## SMDP Discounting

### How do I apply SMDP discounting?

Use `duration` from the info payload:

```python
obs, reward, term, trunc, info = env.step(action)

# Get duration (equals k_exec)
duration = info["smdp"]["duration"]

# Apply SMDP discounting
gamma = 0.99
discounted_reward = reward * (gamma ** duration)

# Track cumulative time for multi-step discounting
total_time += duration
next_discount = gamma ** total_time
```

### What happens with early termination?

When the episode terminates before an option completes, `duration` reflects actual execution:

```python
# Option has 5 actions, but episode terminates after 2 actions

info["smdp"] = {
    "k_exec": 2,              # 2 actions executed
    "duration": 2,            # 2 ticks (= k_exec)
    "terminated_early": True  # Flag indicating early termination
}
```

**Why this matters:**
- Use actual duration for discounting, not planned length
- Prevents incorrect temporal credit assignment
- Handles partial execution correctly

## Action Interfaces

### Index vs Direct: Which should I use?

| Use Case | Recommended Interface | Why |
|----------|----------------------|-----|
| **RL Training** | Index | Discrete action space for algorithms |
| **Scripting/Testing** | Direct | Intuitive option objects |
| **Action Masking** | Index | Built-in mask support |
| **Debugging** | Direct | Clear option identification |
| **Continuous Actions** | Direct | Natural continuous support |

**Common Pattern**: Start with direct for prototyping, switch to index for RL training.

### How does action masking work in index interface?

```python
def availability_fn(obs):
    """Return available option indices."""
    if obs[0] > 0.5:  # Cart far right
        return [0, 2]  # Only left-based options
    else:
        return [0, 1, 2]  # All options

env = SMDPfier(..., availability_fn=availability_fn)

obs, info = env.reset()
mask = info["smdp"]["action_mask"]  # e.g., [1, 0, 1] = options 0,2 available
```

### What happens with dynamic options overflow?

When dynamic options exceed `max_options`:

```python
env = SMDPfier(..., max_options=3)

# If generator returns 5 options:
# - First 3 options are used
# - Last 2 options are dropped  
# - info["smdp"]["num_dropped"] = 2
```

## Environment Compatibility

### Can I use continuous actions?

**Yes!** SMDPfier fully supports continuous action spaces:

```python
# Continuous options
continuous_options = [
    Option([[-1.0], [0.0], [1.0]], "left-center-right"),  # 3 ticks
    Option([[0.5, 0.2]], "multi-dim-action"),             # 1 tick
]

env = SMDPfier(
    gym.make("Pendulum-v1"),
    options_provider=continuous_options,
    action_interface="direct"  # Recommended for continuous
)
```

### What environments work with SMDPfier?

**Any Gymnasium environment!** SMDPfier is a generic wrapper.

✅ **Confirmed Compatible:**
- CartPole, Pendulum, MountainCar
- Atari games
- MuJoCo environments  
- Custom environments
- Multi-agent environments (with appropriate setup)

### What happens if my environment terminates during an option?

SMDPfier handles this gracefully:

1. **Execution stops** immediately when `env.step()` returns `terminated=True` or `truncated=True`
2. **Duration reflects actual execution** (duration = k_exec)
3. **Info payload** contains early termination flag

```python
# Option with 5 actions, but episode terminates after 2 actions
info["smdp"] = {
    "k_exec": 2,                    # Only 2 actions executed
    "duration": 2,                  # 2 ticks (= k_exec)
    "terminated_early": True,       # Flag early termination
    # ...
}
```

## Common Issues

### My RL algorithm isn't learning - what's wrong?

**Common causes:**

1. **Wrong discounting**: Use `duration`, not step count
```python
# Wrong
discount = gamma ** step_count

# Right  
discount = gamma ** info["smdp"]["duration"]
```

2. **Action space mismatch**: Ensure max_options is correct
```python
# Correct for static options
max_options = len(static_options)

# Might be needed for dynamic options
max_options = 10  # Set appropriately
```

3. **Reward aggregation**: Check if default sum is appropriate
```python
# Try different aggregation
reward_agg = mean_rewards          # Average instead of sum
reward_agg = discounted_sum(0.99)  # Internal discounting
```

### I'm getting SMDPOptionValidationError - what does it mean?

This means an option failed precheck validation:

```python
try:
    env.step(action)
except SMDPOptionValidationError as e:
    print(f"Option '{e.option_name}' failed at step {e.failing_step_index}")
    print(f"Action {e.action_repr} is invalid in current state")
    print(f"State summary: {e.short_obs_summary}")
```

**Solutions:**
- Check if actions are valid for current environment state
- Verify action space compatibility  
- Use availability_fn to mask invalid options
- Turn off precheck: `precheck=False`

### My options aren't being executed correctly

**Debug steps:**

1. **Check option construction**:
```python
option = Option([0, 1, 0], "test")
print(f"Actions: {option.actions}")
print(f"Length: {len(option.actions)}")
```

2. **Verify action validity**:
```python
# Test each action individually in base environment
for action in option.actions:
    obs, reward, term, trunc, info = base_env.step(action)
    if term or trunc:
        print(f"Action {action} terminates episode!")
```

3. **Check execution details**:
```python
obs, reward, term, trunc, info = smdp_env.step(option)
smdp = info["smdp"]
print(f"Planned steps: {smdp['option']['len']}")
print(f"Executed steps: {smdp['k_exec']}")
print(f"Rewards: {smdp['rewards']}")
```

### Performance is slow - how can I optimize?

**Performance tips:**

1. **Cache options** when possible:
```python
# Pre-compute static options
static_options = [Option([i, j], f"option_{i}_{j}") 
                 for i in range(2) for j in range(2)]
```

2. **Use simple reward aggregators**:
```python
# Simple aggregators are faster
from smdpfier.defaults import sum_rewards, mean_rewards
reward_agg = sum_rewards  # Fast
```

3. **Use appropriate max_options**:
```python
# Don't over-allocate
max_options = 5   # If you typically have 3-5 options
# vs  
max_options = 100 # Wastes memory and computation
```

## Advanced Usage

### Can I modify options during execution?

**No.** Options are executed atomically. However, you can:

1. **Use dynamic options** that change between executions
2. **Create context-aware generators** that consider current state
3. **Implement early termination** via environment design

### How do I debug option sequences?

**Detailed info inspection:**

```python
obs, reward, term, trunc, info = env.step(action)
smdp = info["smdp"]

print("=== Option Execution Details ===")
print(f"Option: {smdp['option']['name']} (ID: {smdp['option']['id']})")
print(f"Actions: {smdp['option']['len']} total")
print(f"Executed: {smdp['k_exec']} steps")
print(f"Duration: {smdp['duration']} ticks (= k_exec)")
print(f"Per-step rewards: {smdp['rewards']}")
print(f"Macro reward: {reward}")
print(f"Early termination: {smdp['terminated_early']}")

if smdp.get('action_mask'):
    print(f"Available options next: {smdp['action_mask']}")
```

### Can I nest SMDPfiers?

**Technically possible but not recommended.** SMDPfier is designed as a single-level abstraction. For multi-level hierarchy, consider:

1. **Option composition**: Create complex options from simple ones
2. **Custom option generators**: Generate hierarchical option sets
3. **External hierarchy**: Use SMDPfier as one level in a larger system

---

**Still have questions?** Check the [API Reference](api.md) or [examples](../examples/) for more details.
