# API Reference

Complete reference for SMDPfier classes, functions, and configurations.

## Quick Start Examples

### Basic SMDPfier Setup

```python
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration, sum_rewards
import gymnasium as gym

# Basic setup with static options
env = gym.make("CartPole-v1")
options = [Option([0, 1], "left-right"), Option([1, 0], "right-left")]

smdp_env = SMDPfier(
    env,
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="index"
)
```

### Dynamic Options with Built-in Generators

```python
from smdpfier.defaults.options import RandomStaticLen
from smdpfier.defaults.durations import RandomActionDuration

smdp_env = SMDPfier(
    env,
    options_provider=RandomStaticLen(length=3, num_options=8),
    duration_fn=RandomActionDuration(min_duration=2, max_duration=5),
    action_interface="index",
    max_options=8
)
```

## Core Classes

### SMDPfier

**Primary wrapper class that transforms any Gymnasium environment into an SMDP.**

```python
class SMDPfier(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        options_provider: Callable[[Any, dict], list[Option]] | Sequence[Option],
        duration_fn: Callable[[Option, Any, dict], int | list[int]],
        reward_agg: Callable[[list[float]], float] = sum_rewards,
        action_interface: Literal["index", "direct"] = "index",
        max_options: int | None = None,
        availability_fn: Optional[Callable[[Any], Iterable[int]]] = None,
        precheck: bool = False,
        partial_duration_policy: Literal["proportional", "full", "zero"] = "proportional",
        time_units: str = "ticks",
        rng_seed: int | None = None,
    )
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `env` | `gym.Env` | ✅ | - | Base Gymnasium environment |
| `options_provider` | `Callable` or `Sequence[Option]` | ✅ | - | Static options or dynamic generator |
| `duration_fn` | `Callable` | ✅ | - | Returns duration(s) for option |
| `reward_agg` | `Callable` | ❌ | `sum_rewards` | How to aggregate per-step rewards |
| `action_interface` | `"index"` or `"direct"` | ❌ | `"index"` | Action selection interface |
| `max_options` | `int` or `None` | ❌ | `None` | Max options (required for index interface) |
| `availability_fn` | `Callable` or `None` | ❌ | `None` | Action masking function |
| `precheck` | `bool` | ❌ | `False` | Validate options before execution |
| `partial_duration_policy` | `str` | ❌ | `"proportional"` | Handle early termination |
| `time_units` | `str` | ❌ | `"ticks"` | Time unit name (metadata only) |
| `rng_seed` | `int` or `None` | ❌ | `None` | Random seed for reproducibility |

#### Methods

**`step(action: int | Option) -> tuple[obs, reward, terminated, truncated, info]`**

Execute an option in the environment.

- **Index interface**: `action` is integer index
- **Direct interface**: `action` is Option object

**`reset(**kwargs) -> tuple[obs, info]`**

Reset the environment and return initial observation and info.

**`close()`**

Close the environment.

### Option

**Represents a sequence of primitive actions with metadata.**

```python
class Option:
    def __init__(
        self,
        actions: Sequence[Any],
        name: str,
        meta: dict | None = None
    )
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `actions` | `Sequence[Any]` | ✅ | Sequence of primitive actions |
| `name` | `str` | ✅ | Human-readable option name |
| `meta` | `dict` or `None` | ❌ | Additional metadata |

#### Properties

- **`actions`**: The action sequence
- **`name`**: Human-readable name
- **`meta`**: User-defined metadata dictionary
- **`id`**: Stable hash-based identifier (computed from actions + name)

#### Examples

```python
# Discrete actions
option1 = Option([0, 1, 0], "left-right-left")

# Continuous actions
option2 = Option([[-1.0], [0.5], [2.0]], "continuous-sequence")

# With metadata
option3 = Option(
    actions=[0, 0, 1, 1], 
    name="double-pairs",
    meta={"category": "symmetric", "difficulty": "easy"}
)

print(option1.id)  # Stable hash: "a1b2c3..."
```

## SMDP Info Payload Structure

Every `step()` call returns comprehensive metadata in `info["smdp"]`:

```python
{
    "option": {
        "id": "abc123...",              # Stable hash-based ID
        "name": "left-right-left",      # Human-readable name
        "len": 3,                       # Number of primitive actions
        "meta": {"category": "test"}    # User metadata (if any)
    },
    "k_exec": 3,                        # Primitive steps actually executed
    "rewards": [1.0, 1.0, 1.0],        # Per-step rewards
    "duration_planned": 10,             # Expected duration (ticks)
    "duration_exec": 10,                # Actual duration (accounting for early termination)
    "terminated_early": False,          # Whether episode ended during option
    "time_units": "ticks",              # Always "ticks"
    "action_mask": [1, 1, 0, 1],       # Available option indices (index interface only)
    "num_dropped": 0                    # Options dropped due to overflow (index interface only)
}
```

### Info Fields Detail

| Field | Type | Description |
|-------|------|-------------|
| `option.id` | `str` | Stable identifier (hash of actions + name) |
| `option.name` | `str` | Human-readable option name |
| `option.len` | `int` | Total number of primitive actions |
| `option.meta` | `dict` | User-defined metadata |
| `k_exec` | `int` | Number of primitive steps actually executed |
| `rewards` | `list[float]` | Per-primitive-step rewards |
| `duration_planned` | `int` | Expected duration from duration function |
| `duration_exec` | `int` | Actual duration (may differ due to early termination) |
| `terminated_early` | `bool` | True if episode ended before option completed |
| `time_units` | `str` | Always "ticks" |
| `action_mask` | `list[int]` | Binary mask of available options (index interface) |
| `num_dropped` | `int` | Number of options dropped due to overflow (index interface) |

## Action Interfaces

### Index Interface

Transforms the action space to `Discrete(max_options)` where actions are integer indices.

**Configuration:**
```python
env = SMDPfier(
    base_env,
    options_provider=options,
    duration_fn=duration_fn,
    action_interface="index",
    max_options=len(options)  # Required
)

# Usage
action = 1  # Select second option
obs, reward, term, trunc, info = env.step(action)
```

**Features:**
- Built-in action masking via `info["smdp"]["action_mask"]`
- Overflow handling for dynamic options
- Seamless RL algorithm integration

### Direct Interface

Allows passing `Option` objects directly to `step()`.

**Configuration:**
```python
env = SMDPfier(
    base_env,
    options_provider=options,
    duration_fn=duration_fn,
    action_interface="direct"
    # No max_options needed
)

# Usage
option = options[1]  # Select option object
obs, reward, term, trunc, info = env.step(option)
```

**Features:**
- Intuitive option selection
- Full control over option choice
- Works naturally with continuous actions

## Built-in Defaults

### Option Generators

**`RandomStaticLen`** - Generate random options with fixed length:

```python
from smdpfier.defaults.options import RandomStaticLen

generator = RandomStaticLen(
    length=3,                    # Fixed option length
    action_space_size=4,         # Discrete action space size (auto-detected if None)
    num_options=10,              # Number of options to generate
    rng_seed=42                  # Random seed
)
```

**`RandomVarLen`** - Generate random options with variable length:

```python
from smdpfier.defaults.options import RandomVarLen

generator = RandomVarLen(
    min_length=2,                # Minimum option length
    max_length=5,                # Maximum option length
    action_space_size=4,         # Discrete action space size
    num_options=8,               # Number of options to generate
    rng_seed=42                  # Random seed
)
```

### Duration Functions

**Option-Level Durations (Scalar):**

```python
from smdpfier.defaults.durations import ConstantOptionDuration, RandomOptionDuration

# Fixed duration per option
ConstantOptionDuration(10)  # Every option takes 10 ticks

# Random duration per option
RandomOptionDuration(min_duration=5, max_duration=15, rng_seed=42)
```

**Action-Level Durations (List):**

```python
from smdpfier.defaults.durations import ConstantActionDuration, RandomActionDuration

# Fixed duration per action
ConstantActionDuration(3)  # Each action takes 3 ticks

# Random duration per action
RandomActionDuration(min_duration=2, max_duration=5, rng_seed=42)
```

**Custom Duration Mapping:**

```python
from smdpfier.defaults.durations import MapActionDuration

# Map specific actions to durations
action_durations = {0: 2, 1: 5, 2: 3}  # Action 0: 2 ticks, Action 1: 5 ticks, etc.
MapActionDuration(action_durations, default_duration=4)
```

### Reward Aggregation

```python
from smdpfier.defaults.rewards import sum_rewards, mean_rewards, discounted_sum

# Sum all per-step rewards (default)
reward_agg = sum_rewards

# Average per-step rewards
reward_agg = mean_rewards

# Discount per-step rewards with γ
reward_agg = discounted_sum(gamma=0.99)
```

## Configuration Patterns

### Static Options with Index Interface

```python
options = [
    Option([0, 1], "left-right"),
    Option([1, 0], "right-left"),
    Option([0, 0, 1], "left-left-right"),
]

env = SMDPfier(
    base_env,
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="index",
    max_options=len(options)
)
```

### Dynamic Options with Masking

```python
def dynamic_generator(obs, info):
    # Generate options based on current state
    if obs[0] > 0:
        return [Option([0], "left"), Option([0, 0], "double-left")]
    else:
        return [Option([1], "right"), Option([1, 1], "double-right")]

def availability_mask(obs):
    # Mask options based on state
    return [0, 1] if obs[1] > 0.5 else [0, 1]  # All available

env = SMDPfier(
    base_env,
    options_provider=dynamic_generator,
    duration_fn=RandomActionDuration(2, 4),
    action_interface="index",
    max_options=5,
    availability_fn=availability_mask
)
```

### Continuous Actions with Direct Interface

```python
continuous_options = [
    Option([[-1.0], [0.0]], "left-center"),
    Option([[0.5], [1.0]], "gentle-hard-right"),
    Option([[-2.0], [2.0], [0.0]], "extreme-swing"),
]

env = SMDPfier(
    gym.make("Pendulum-v1"),
    options_provider=continuous_options,
    duration_fn=ConstantActionDuration(3),
    action_interface="direct"
)
```

## Partial Duration Policies

Handle early termination when episodes end before options complete:

| Policy | Behavior | Formula | Use Case |
|--------|----------|---------|----------|
| `"proportional"` | Scale by execution ratio | `(k_exec / option_len) * planned_duration` | **Default** - realistic time scaling |
| `"full"` | Use full planned duration | `planned_duration` | Options have setup costs |
| `"zero"` | No time consumed | `0` | Failed options waste no time |

**Example:**
```python
# Option with 4 actions, 12 ticks planned, terminates after 3 actions

# Proportional (default): (3/4) * 12 = 9 ticks
# Full: 12 ticks  
# Zero: 0 ticks

env = SMDPfier(
    base_env,
    options_provider=options,
    duration_fn=ConstantOptionDuration(12),
    partial_duration_policy="proportional"  # Default
)
```

## Error Handling

SMDPfier provides detailed error context through specialized exceptions:

```python
from smdpfier.errors import SMDPOptionValidationError, SMDPOptionExecutionError

try:
    obs, reward, term, trunc, info = env.step(action)
except SMDPOptionValidationError as e:
    print(f"Precheck failed for option '{e.option_name}' at step {e.failing_step_index}")
    print(f"Action: {e.action_repr}, State: {e.short_obs_summary}")
except SMDPOptionExecutionError as e:
    print(f"Runtime error for option '{e.option_name}' at step {e.failing_step_index}")
    print(f"Underlying error: {e.base_error}")
```

See [Error Handling](errors.md) for complete details.

## Custom Functions

### Custom Options Provider

```python
def custom_options_provider(obs, info):
    """Generate options based on observation and info."""
    # Access current state
    position = obs[0]
    
    # Access action space if needed
    action_space = info.get("action_space")
    
    # Access action mask if available
    action_mask = info.get("action_mask")
    
    # Generate options
    options = []
    if position > 0:
        options.append(Option([0, 0], "strong-left"))
    if position < 0:
        options.append(Option([1, 1], "strong-right"))
    
    return options
```

### Custom Duration Function

```python
def custom_duration_fn(option, obs, info):
    """Compute duration based on option and state."""
    base_duration = len(option.actions) * 2
    
    # State-dependent adjustment
    if obs[0] > 0.5:  # Far right position
        return base_duration + 3  # Takes longer
    else:
        return base_duration

# Can return scalar (int) or list (list[int])
def per_action_duration_fn(option, obs, info):
    """Return duration for each action."""
    durations = []
    for action in option.actions:
        if action == 0:  # Left action
            durations.append(2)
        else:  # Right action
            durations.append(5)
    return durations
```

### Custom Availability Function

```python
def custom_availability_fn(obs):
    """Return available option indices based on state."""
    position, velocity = obs[0], obs[1]
    
    available = []
    
    # Always allow basic options
    available.extend([0, 1])
    
    # Complex options only when stable
    if abs(velocity) < 0.1:
        available.extend([2, 3, 4])
    
    return available
```

## Performance Tips

### Efficient Option Generation

```python
# Pre-compute static options when possible
static_options = [Option([0, 1], f"option_{i}") for i in range(10)]

# Cache dynamic options when state doesn't change much
class CachedOptionsProvider:
    def __init__(self):
        self._cache = {}
    
    def __call__(self, obs, info):
        state_key = tuple(obs[:2])  # Use subset of observation as key
        if state_key not in self._cache:
            self._cache[state_key] = generate_options_for_state(obs)
        return self._cache[state_key]
```

### Memory-Efficient Duration Functions

```python
# Use generators for large option sets
def memory_efficient_duration_fn(option, obs, info):
    # Compute duration on demand rather than storing
    return len(option.actions) * compute_action_cost(obs)
```

## Action Masking

Action masking in SMDPfier is handled through the `availability_fn` parameter and works exclusively with the index interface. See the [Masking and Precheck](masking_and_precheck.md) guide for comprehensive examples.

```python
def availability_fn(obs):
    """Return list of available option indices."""
    # Return indices of valid options based on current state
    return [0, 2, 3]  # Options 1 is masked out

env = SMDPfier(
    base_env,
    options_provider=options,
    duration_fn=duration_fn,
    action_interface="index",
    availability_fn=availability_fn
)

# Action mask appears in info
obs, info = env.reset()
mask = info["smdp"]["action_mask"]  # e.g., [1, 0, 1, 1]
```

---

**See Also:**
- [Duration Guide](durations.md) - Understanding ticks and SMDP discounting
- [Interface Guide](usage_index_vs_direct.md) - Choosing index vs direct
- [Error Handling](errors.md) - Debugging failed options
- [Examples](../examples/) - Complete working examples
