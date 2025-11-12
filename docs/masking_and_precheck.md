# Masking and Precheck

SMDPfier provides sophisticated action masking and precheck validation to handle invalid options gracefully and ensure robust option execution.

## Action Masking

Action masking restricts which options are available based on the current environment state. This is essential for handling **state-dependent action validity** in complex environments.

### Key Concepts

- **Availability Function**: Determines which option indices are valid
- **Action Mask**: Binary array indicating available options  
- **Index Interface Only**: Masking only works with `action_interface="index"`
- **Dynamic Evaluation**: Mask is recomputed every step

### Basic Action Masking

```python
import gymnasium as gym
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration

# Define options for CartPole
options = [
    Option([0, 0], "strong-left"),      # Index 0
    Option([1, 1], "strong-right"),     # Index 1  
    Option([0, 1], "left-right"),       # Index 2
    Option([1, 0], "right-left"),       # Index 3
]

def cart_availability(obs):
    """Restrict options based on cart position and velocity."""
    position, velocity = obs[0], obs[1]
    
    available = []
    
    # Always allow balanced options
    available.extend([2, 3])  # left-right, right-left
    
    # Restrict strong movements based on position
    if position < 0.3:    # Cart not too far right
        available.append(1)   # Allow strong-right
    if position > -0.3:   # Cart not too far left  
        available.append(0)   # Allow strong-left
        
    return available

env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="index",
    max_options=4,
    availability_fn=cart_availability
)

obs, info = env.reset()
print(f"Available options: {info['smdp']['action_mask']}")
# Might show: [1, 1, 1, 1] (all available) or [0, 1, 1, 1] (strong-left masked)
```

### Action Mask Structure

The action mask is a binary list where `1` means available and `0` means masked:

```python
action_mask = [1, 0, 1, 0]  # Options 0 and 2 available, 1 and 3 masked
```

**Usage in RL algorithms:**

```python
obs, info = env.reset()
action_mask = info["smdp"]["action_mask"]

# In your RL algorithm:
if action_mask is not None:
    # Mask invalid actions (set their Q-values to -inf)
    masked_q_values = q_values.copy()
    masked_q_values[action_mask == 0] = -float('inf')
    action = np.argmax(masked_q_values)
else:
    action = np.argmax(q_values)
```

### Complex Masking Examples

#### Environment-Specific Masking

```python
def taxi_availability(obs):
    """Taxi-v3 environment masking."""
    # Decode Taxi state
    taxi_row, taxi_col, passenger_loc, destination = env.unwrapped.decode(obs)
    
    available = []
    
    # Movement actions (always available)
    available.extend([0, 1, 2, 3])  # south, north, east, west
    
    # Pickup action (only if passenger at taxi location)
    if passenger_loc < 4:  # Passenger not in taxi
        passenger_coords = env.unwrapped.locs[passenger_loc]
        if (taxi_row, taxi_col) == passenger_coords:
            available.append(4)  # Allow pickup
    
    # Dropoff action (only if passenger in taxi at destination)
    if passenger_loc == 4:  # Passenger in taxi
        destination_coords = env.unwrapped.locs[destination]
        if (taxi_row, taxi_col) == destination_coords:
            available.append(5)  # Allow dropoff
            
    return available

# Taxi options
taxi_options = [
    Option([0], "south"),      # Index 0
    Option([1], "north"),      # Index 1
    Option([2], "east"),       # Index 2
    Option([3], "west"),       # Index 3
    Option([4], "pickup"),     # Index 4
    Option([5], "dropoff"),    # Index 5
]

taxi_env = SMDPfier(
    gym.make("Taxi-v3"),
    options_provider=taxi_options,
    duration_fn=ConstantOptionDuration(1),
    action_interface="index",
    max_options=6,
    availability_fn=taxi_availability
)
```

#### State-Dependent Option Length

```python
def adaptive_availability(obs):
    """Allow different option lengths based on state."""
    velocity = abs(obs[1])  # Cart velocity
    
    if velocity > 0.5:  # High velocity - need quick corrections
        return [0, 1]   # Only single-action options
    else:  # Low velocity - can use longer sequences
        return [0, 1, 2, 3, 4]  # All options available

options = [
    Option([0], "quick-left"),        # Index 0 - quick
    Option([1], "quick-right"),       # Index 1 - quick  
    Option([0, 0], "double-left"),    # Index 2 - longer
    Option([1, 1], "double-right"),   # Index 3 - longer
    Option([0, 1, 0], "zigzag"),      # Index 4 - longest
]

env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=options,
    duration_fn=ConstantOptionDuration(3),
    action_interface="index",
    max_options=5,
    availability_fn=adaptive_availability
)
```

### Dynamic Options with Masking

When using dynamic option generators, the `availability_fn` is automatically passed to restrict generated options:

```python
from smdpfier.defaults.options import RandomStaticLen

def state_aware_generator(obs, info):
    """Generate options based on state, using availability info."""
    # Get action mask from info (passed automatically)
    action_mask = info.get("action_mask")
    
    if action_mask is not None:
        # Generate options only with available actions
        available_actions = [i for i, avail in enumerate(action_mask) if avail]
    else:
        # No masking - use all actions
        available_actions = list(range(info["action_space"].n))
    
    # Generate random options with available actions only
    options = []
    for i in range(5):
        if available_actions:
            actions = random.choices(available_actions, k=3)
            options.append(Option(actions, f"dynamic_{i}"))
    
    return options

def base_availability(obs):
    """Base availability function."""
    if obs[0] > 0:  # Cart right
        return [0]  # Only left action
    else:
        return [0, 1]  # Both actions

env = SMDPfier(
    gym.make("CartPole-v1"),
    options_provider=state_aware_generator,
    duration_fn=ConstantOptionDuration(2),
    action_interface="index",
    max_options=5,
    availability_fn=base_availability  # Passed to generator automatically
)
```

## Precheck Validation

Precheck validation attempts to validate options **before execution** by testing their actions in the current environment state.

### Enabling Precheck

```python
env = SMDPfier(
    base_env,
    options_provider=options,
    duration_fn=ConstantOptionDuration(5),
    action_interface="index", 
    precheck=True  # Enable precheck validation
)
```

### How Precheck Works

1. **Before executing an option**, SMDPfier saves the environment state
2. **Tests each action** in the option sequence
3. **Restores the environment state** after testing
4. **Raises SMDPOptionValidationError** if any action fails
5. **Proceeds with execution** if all actions are valid

### Precheck Example

```python
from smdpfier.errors import SMDPOptionValidationError

# Option that might be invalid in some states
risky_option = Option([0, 1, 0, 1, 0], "risky-sequence")

try:
    obs, reward, term, trunc, info = env.step(risky_option)
except SMDPOptionValidationError as e:
    print(f"Option '{e.option_name}' failed precheck!")
    print(f"Failed at step {e.failing_step_index}")
    print(f"Action {e.action_repr} is invalid")
    print(f"Environment state: {e.short_obs_summary}")
    
    # Handle the error (e.g., try a different option)
    fallback_option = Option([0], "safe-fallback")
    obs, reward, term, trunc, info = env.step(fallback_option)
```

### Precheck Limitations

⚠️ **Important Limitations:**

1. **Environment must support state save/restore** (not all environments do)
2. **Performance overhead** from testing each option
3. **May not catch all edge cases** (e.g., stochastic environments)
4. **False positives** possible in complex environments

**Recommendation**: Use precheck during **development and debugging**, consider disabling in **production** for performance.

### Precheck vs Masking

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Action Masking** | Known invalid patterns | Fast, reliable | Requires domain knowledge |
| **Precheck** | Unknown failure modes | Automatic detection | Slower, may have false positives |
| **Both** | Maximum safety | Comprehensive validation | Higher complexity |

## Best Practices

### Masking Strategy

```python
def robust_availability(obs):
    """Comprehensive availability function."""
    available = []
    
    # Conservative base set (always safe)
    available.extend([0, 1])  # Basic actions
    
    # Add options based on state confidence
    confidence = compute_state_confidence(obs)
    
    if confidence > 0.8:
        available.extend([2, 3])  # Medium complexity options
    
    if confidence > 0.9:
        available.extend([4, 5, 6])  # High complexity options
    
    return available
```

### Error-Resilient Option Design

```python
# Design options to minimize failure probability
safe_options = [
    Option([0], "single-left"),       # Minimal option - rarely fails
    Option([1], "single-right"),      # Minimal option - rarely fails
    Option([0, 1], "balanced"),       # Balanced - self-correcting
]

# Avoid overly long or extreme options  
risky_options = [
    Option([0]*10, "extreme-left"),   # Long sequence - high failure risk
    Option([1]*10, "extreme-right"),  # Long sequence - high failure risk
]
```

### Integration with RL Algorithms

```python
class MaskedDQNAgent:
    def select_action(self, obs, info):
        q_values = self.q_network(obs)
        
        # Apply action mask if available
        action_mask = info.get("smdp", {}).get("action_mask")
        if action_mask is not None:
            masked_q_values = q_values.copy()
            masked_q_values[np.array(action_mask) == 0] = -float('inf')
            return np.argmax(masked_q_values)
        else:
            return np.argmax(q_values)
    
    def train(self, env):
        obs, info = env.reset()
        
        while True:
            action = self.select_action(obs, info)
            
            try:
                obs, reward, term, trunc, info = env.step(action)
                # ... training logic ...
            except SMDPOptionValidationError:
                # Handle validation failure
                continue
                
            if term or trunc:
                break
```

## Debugging Masking Issues

### Inspect Masking Behavior

```python
def debug_masking(env, num_steps=10):
    """Debug action masking behavior."""
    obs, info = env.reset()
    
    for step in range(num_steps):
        mask = info.get("smdp", {}).get("action_mask", [])
        available_actions = [i for i, avail in enumerate(mask) if avail == 1]
        
        print(f"Step {step}:")
        print(f"  Observation: {obs[:3]}...")  # First 3 elements
        print(f"  Action mask: {mask}")
        print(f"  Available actions: {available_actions}")
        
        if available_actions:
            action = random.choice(available_actions)
            obs, reward, term, trunc, info = env.step(action)
            
            if term or trunc:
                break
        else:
            print("  No actions available!")
            break

debug_masking(env)
```

### Common Masking Issues

**Issue: All actions masked**
```python
def overly_restrictive_availability(obs):
    if obs[0] > 2.0:  # Impossible condition
        return []  # No actions available!
    return [0, 1]

# Fix: Ensure at least one action is always available
def better_availability(obs):
    available = [0]  # Always allow basic action
    if obs[0] < 2.0:
        available.append(1)
    return available
```

**Issue: Inconsistent masking**
```python
def inconsistent_availability(obs):
    # Problem: Random masking
    return random.choices([0, 1, 2], k=random.randint(1, 3))

# Fix: Deterministic masking based on state
def consistent_availability(obs):
    position = obs[0]
    if position > 0.5:
        return [0, 2]  # Deterministic based on position
    else:
        return [1, 2]
```

## Summary

| Feature | Purpose | Best For |
|---------|---------|----------|
| **Action Masking** | Restrict invalid options | State-dependent validity |
| **Precheck** | Test options before execution | Unknown failure modes |
| **Combined** | Maximum robustness | Critical applications |

**Key Takeaways:**
- Use masking for **known patterns** of invalid actions
- Use precheck for **unknown failure modes** during development
- Design options to **minimize failure probability**
- Always ensure **at least one action remains available**

---

**Next**: [Error Handling](errors.md) | **See Also**: [API Reference](api.md)
