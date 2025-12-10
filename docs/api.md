# API Reference

Complete reference for SMDPfier classes, functions, and configurations.

## Quick Start Examples

### Basic SMDPfier Setup

```python
from smdpfier import SMDPfier, Option
from smdpfier.defaults import sum_rewards
import gymnasium as gym

# Basic setup with static options
env = gym.make("CartPole-v1")
options = [
    Option([0, 1], "left-right"),     # 2 actions = 2 ticks
    Option([1, 0], "right-left")      # 2 actions = 2 ticks
]

smdp_env = SMDPfier(
    env,
    options_provider=options,
    action_interface="index",
    max_options=len(options)
)
```

### Dynamic Options with Built-in Generators

```python
from smdpfier.defaults.options import RandomStaticLen

smdp_env = SMDPfier(
    env,
    options_provider=RandomStaticLen(length=3, num_options=8),
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
        reward_agg: Callable[[list[float]], float] = sum_rewards,
        action_interface: Literal["index", "direct"] = "index",
        max_options: int | None = None,
        availability_fn: Optional[Callable[[Any], Iterable[int]]] = None,
        precheck: bool = False,
        rng_seed: int | None = None,
    )
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `env` | `gym.Env` | ✅ | - | Base Gymnasium environment |
| `options_provider` | `Callable` or `Sequence[Option]` | ✅ | - | Static options or dynamic generator |
| `reward_agg` | `Callable` | ❌ | `sum_rewards` | How to aggregate per-step rewards |
| `action_interface` | `"index"` or `"direct"` | ❌ | `"index"` | Action selection interface |
| `max_options` | `int` or `None` | ❌ | `None` | Max options (required for index interface) |
| `availability_fn` | `Callable` or `None` | ❌ | `None` | Action masking function |
| `precheck` | `bool` | ❌ | `False` | Validate options before execution |
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

### Option (Abstract Base Class)

**Abstract base class defining the stateful execution interface for temporal abstractions.**

Options represent sequences of actions that can span multiple time steps. The stateful interface allows options to observe the environment and adapt their behavior dynamically.

#### Stateful Execution Lifecycle

```
1. option.begin(obs, info)           # Initialize state
2. Loop:
   a. action, done = option.act(obs, info)  # Choose next action
   b. If action is None: break       # Option terminates without action
   c. obs, r, term, trunc, info = env.step(action)
   d. option.on_step(obs, r, term, trunc, info)  # Process result
   e. If done or term or trunc: break
3. Return aggregated reward and duration=k_exec
```

#### Abstract Methods

All custom options must implement these methods:

```python
from abc import ABC, abstractmethod
from smdpfier.option import OptionBase

class MyOption(OptionBase):
    @abstractmethod
    def begin(self, obs: Any, info: dict[str, Any]) -> None:
        """Initialize option state before execution.
        
        Called once at the start with initial observation.
        Use this to reset counters, initialize policies, etc.
        """
        pass
    
    @abstractmethod
    def act(self, obs: Any, info: dict[str, Any]) -> Any | tuple[Any, bool]:
        """Select next action based on current observation.
        
        Returns:
            - action: Next action to execute
            - (action, done): Action and termination flag
            - None or (None, True): Terminate without executing action
        
        The option can observe current state and adapt dynamically.
        Return done=True to signal early termination.
        """
        pass
    
    @abstractmethod
    def on_step(self, obs: Any, reward: float, terminated: bool, 
                truncated: bool, info: dict[str, Any]) -> None:
        """Process the result of executing an action.
        
        Called after each env.step() with the transition data.
        Use this to update internal state, collect statistics, etc.
        """
        pass
    
    @abstractmethod
    def preview(self, obs: Any, info: dict[str, Any]) -> Any | None:
        """Preview the first action without executing.
        
        Used for action masking in discrete environments.
        
        Returns:
            - First action that would be executed
            - None if option cannot determine or is unavailable
        """
        pass
    
    @abstractmethod
    def identity(self) -> tuple[str, ...]:
        """Return stable identity tuple for hashing.
        
        Used to generate stable option IDs across equivalent instances.
        
        Returns:
            Tuple of strings uniquely identifying this option's behavior.
        """
        pass
```

#### Properties

All options have these properties:

- **`option_id`**: Stable hash-based identifier (from `identity()`)
- **`name`**: Human-readable name (overridable property)
- **`meta`**: Optional metadata dictionary (overridable property)

### ListOption (Concrete Implementation)

**Concrete option that executes a fixed list of actions.**

This is the most common option type and provides backward compatibility with the original Option dataclass.

```python
from smdpfier import ListOption

option = ListOption(
    actions=[0, 1, 0],
    _name="left-right-left",
    _meta={"category": "basic"}
)
```

For convenience, use the `Option()` factory function:

```python
from smdpfier import Option

# Simple usage (recommended)
option = Option([0, 1, 0], "left-right-left")

# With metadata
option = Option([0, 0, 1, 1], "double-pairs", 
                meta={"category": "symmetric"})
```

#### ListOption Examples

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

print(option1.option_id)  # Stable hash: "a1b2c3..."
print(len(option1))       # 3 (number of actions)
```

### Custom Stateful Options

Create options that adapt to observations:

```python
from smdpfier.option import OptionBase

class ThresholdOption(OptionBase):
    """Execute action 0 until observation exceeds threshold."""
    
    def __init__(self, threshold: float = 0.0, max_steps: int = 5):
        self.threshold = threshold
        self.max_steps = max_steps
        self.step_count = 0
    
    def begin(self, obs, info):
        """Reset step counter."""
        self.step_count = 0
    
    def act(self, obs, info):
        """Choose action based on observation."""
        if obs[0] > self.threshold:
            return None  # Terminate without action
        
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        return 0, done
    
    def on_step(self, obs, reward, terminated, truncated, info):
        """Track execution (already counted in act)."""
        pass
    
    def preview(self, obs, info):
        """Preview first action."""
        return None if obs[0] > self.threshold else 0
    
    def identity(self):
        """Stable identity for hashing."""
        return ("ThresholdOption", str(self.threshold), str(self.max_steps))
    
    @property
    def name(self):
        return f"threshold_{self.threshold}"

# Usage
option = ThresholdOption(threshold=0.5, max_steps=3)
env = SMDPfier(base_env, options_provider=[option])
```

#### Termination Without Action

Options can terminate immediately without executing any action by returning `None`:

```python
class ConditionalOption(OptionBase):
    def act(self, obs, info):
        if not self.precondition_met(obs):
            return None  # Terminate without acting
        return self.choose_action(obs)
```

When `None` is returned:
- No action is executed in the environment
- `k_exec = 0` and `duration = 0`
- `rewards = []` (empty)
- Option completes immediately

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
    "duration": 3,                      # Duration in ticks (= k_exec)
    "rewards": [1.0, 1.0, 1.0],        # Per-step rewards
    "terminated_early": False,          # Whether episode ended during option
    "action_mask": [1, 1, 0, 1],       # Available option indices (index interface only)
    "num_dropped": 0                    # Options dropped due to overflow (index interface only)
}
```

### Info Fields Detail

| Field | Type | Description |
|-------|------|-------------|
| `option.id` | `str` | Stable identifier (hash of actions + name) |
| `option.name` | `str` | Human-readable option name |
| `option.len` | `int` | Total number of primitive actions in option |
| `option.meta` | `dict` | User-defined metadata |
| `k_exec` | `int` | Number of primitive steps actually executed |
| `duration` | `int` | Duration in ticks (always equals k_exec) |
| `rewards` | `list[float]` | Per-primitive-step rewards |
| `terminated_early` | `bool` | True if episode ended before option completed |
| `action_mask` | `list[int]` | Binary mask of available options (index interface) |
| `num_dropped` | `int` | Number of options dropped due to overflow (index interface) |

## Action Interfaces

### Index Interface
## Action Interfaces

### Index Interface

Transforms the action space to `Discrete(max_options)` where actions are integer indices.

**Configuration:**
```python
env = SMDPfier(
    base_env,
    options_provider=options,
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
    length=3,                    # Fixed option length (= 3 ticks)
    action_space_size=4,         # Discrete action space size (auto-detected if None)
    num_options=10,              # Number of options to generate
    rng_seed=42                  # Random seed
)
```

**`RandomVarLen`** - Generate random options with variable length:

```python
from smdpfier.defaults.options import RandomVarLen

generator = RandomVarLen(
    min_length=2,                # Minimum option length (= 2 ticks)
    max_length=5,                # Maximum option length (= 5 ticks)
    action_space_size=4,         # Discrete action space size
    num_options=8,               # Number of options to generate
    rng_seed=42                  # Random seed
)
```

### Reward Aggregation

```python
from smdpfier.defaults import sum_rewards, mean_rewards, discounted_sum

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
    Option([0, 1], "left-right"),                # 2 ticks
    Option([1, 0], "right-left"),                # 2 ticks
    Option([0, 0, 1], "left-left-right"),        # 3 ticks
]

env = SMDPfier(
    base_env,
    options_provider=options,
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
    action_interface="index",
    max_options=5,
    availability_fn=availability_mask
)
```

### Continuous Actions with Direct Interface

```python
continuous_options = [
    Option([[-1.0], [0.0]], "left-center"),            # 2 ticks
    Option([[0.5], [1.0]], "gentle-hard-right"),       # 2 ticks
    Option([[-2.0], [2.0], [0.0]], "extreme-swing"),   # 3 ticks
]

env = SMDPfier(
    gym.make("Pendulum-v1"),
    options_provider=continuous_options,
    action_interface="direct"
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
