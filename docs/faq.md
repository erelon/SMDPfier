# Frequently Asked Questions

Common questions and answers about SMDPfier concepts, usage, and troubleshooting.

## Core Concepts

### What is SMDPfier?

SMDPfier is a Gymnasium wrapper that transforms any environment into a **Semi-Markov Decision Process (SMDP)** by enabling:
- **Options**: Sequences of primitive actions executed atomically
- **Duration Metadata**: Abstract "ticks" for proper temporal discounting
- **SMDP Discounting**: Using γ^{ticks} instead of γ^{steps}

**Key Insight**: SMDPfier separates execution (environment steps) from time (discounting metadata).

### How is this different from other hierarchical RL libraries?

| Aspect | SMDPfier | Other Libraries |
|--------|----------|-----------------|
| **Focus** | Duration-aware SMDP behavior | General hierarchical RL |
| **Complexity** | Single wrapper class | Full frameworks |
| **Environment Support** | Any Gymnasium env unchanged | Often require specific environments |
| **Temporal Modeling** | Explicit duration metadata | Usually step-based |
| **Integration** | Drop-in wrapper | Framework-specific |

### When should I use SMDPfier?

✅ **Use SMDPfier when you want to:**
- Apply **true SMDP discounting** with γ^{duration}
- Test **hierarchical policies** with temporal abstractions
- Add **option-level control** to existing environments
- **Research temporal abstractions** without framework complexity

❌ **Don't use SMDPfier when:**
- You need complex option discovery algorithms
- You want full hierarchical RL frameworks (use HRL libraries)
- You don't care about temporal discounting (standard MDP is fine)

## The Critical Distinction: Steps vs Duration

### 🚨 Do durations control how many steps are executed?

**NO!** This is the most common misconception.

- **Steps**: Determined by option length, controls `env.step()` calls
- **Duration**: Metadata only, used for SMDP discounting

```python
option = Option([0, 1, 0], "three-actions")     # Always 3 steps
duration_fn = ConstantOptionDuration(100)       # 100 ticks metadata

# Result: 3 env.step() calls, 100 ticks for discounting
```

### What determines the number of environment steps?

**Only the option length** (`len(option.actions)`).

```python
# These all execute the same number of steps
Option([0, 1, 0], "option-1")     # 3 steps
Option([1, 0, 1], "option-2")     # 3 steps  
Option([0, 0, 0], "option-3")     # 3 steps

# Duration function doesn't matter for execution:
ConstantOptionDuration(1)         # Still 3 steps each
ConstantOptionDuration(1000)      # Still 3 steps each
```

### How do I control execution length?

**Change the option's action sequence**, not the duration function:

```python
# Want longer execution? Add more actions:
short_option = Option([0, 1], "short")           # 2 steps
long_option = Option([0, 1, 0, 1, 0], "long")    # 5 steps

# Want shorter execution? Remove actions:
very_short_option = Option([0], "single")        # 1 step
```

## SMDP Discounting

### How do I apply SMDP discounting?

Use `duration_exec` from the info payload:

```python
obs, reward, term, trunc, info = env.step(action)

# Get executed duration (handles early termination)  
duration = info["smdp"]["duration_exec"]

# Apply SMDP discounting
gamma = 0.99
discounted_reward = reward * (gamma ** duration)

# Track cumulative time for multi-step discounting
total_time += duration
next_discount = gamma ** total_time
```

### What's the difference between duration_planned and duration_exec?

| Field | When They Differ | Example |
|-------|-----------------|---------|
| `duration_planned` | Expected duration from duration function | 10 ticks |
| `duration_exec` | Actual duration (accounts for early termination) | 6 ticks (if terminated early) |

**They're equal** when the option completes normally.
**They differ** when the episode terminates before the option finishes.

### How do partial duration policies work?

When an option terminates early:

```python
# Option: [0, 1, 0, 1] (4 actions), Duration: 12 ticks, Terminated after 3 actions

# "proportional" (default): (3/4) * 12 = 9 ticks
# "full": 12 ticks (option conceptually "completes")  
# "zero": 0 ticks (failed options consume no time)
```

Choose based on your domain:
- **Proportional**: Most realistic (default)
- **Full**: Options have setup costs
- **Zero**: Failures are "free"

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
    Option([[-1.0], [0.0], [1.0]], "left-center-right"),
    Option([[0.5, 0.2]], "multi-dim-action"),
]

env = SMDPfier(
    gym.make("Pendulum-v1"),
    options_provider=continuous_options,
    duration_fn=ConstantActionDuration(3),
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
2. **Partial duration policy** is applied based on configuration
3. **Info payload** reflects actual execution (`k_exec`, `duration_exec`, `terminated_early`)

```python
# Option with 5 actions, but episode terminates after 2 actions
info["smdp"] = {
    "k_exec": 2,                    # Only 2 actions executed
    "terminated_early": True,       # Flag early termination
    "duration_exec": 4,             # Partial duration (if proportional policy)
    # ...
}
```

## Common Issues

### My RL algorithm isn't learning - what's wrong?

**Common causes:**

1. **Wrong discounting**: Use `duration_exec`, not step count
```python
# Wrong
discount = gamma ** step_count

# Right  
discount = gamma ** info["smdp"]["duration_exec"]
```

2. **Action space mismatch**: Ensure max_options is correct
```python
# Wrong - action space is Discrete(max_options), not len(options)
max_options = len(static_options)  # Correct for static options
max_options = 10                   # Might be needed for dynamic options
```

3. **Reward aggregation**: Check if `sum_rewards` is appropriate
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

2. **Avoid complex duration functions**:
```python
# Simple is better
ConstantOptionDuration(5)          # Fast
# vs
complex_state_dependent_duration   # Potentially slow
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
print(f"Actions: {len(smdp['option']['len'])} total")
print(f"Executed: {smdp['k_exec']} steps")
print(f"Per-step rewards: {smdp['rewards']}")
print(f"Duration: {smdp['duration_exec']} ticks")
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
