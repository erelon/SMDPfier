# Error Handling

SMDPfier provides comprehensive error handling with detailed context information to help debug option execution issues and build robust SMDP applications.

## Error Types

SMDPfier defines specialized exceptions that provide rich context about option failures, making debugging much easier than generic exceptions.

### SMDPOptionValidationError

Raised during **precheck validation** when an option is determined to be invalid before execution.

```python
from smdpfier.errors import SMDPOptionValidationError

try:
    obs, reward, term, trunc, info = env.step(invalid_option_index)
except SMDPOptionValidationError as e:
    print(f"Validation failed for option '{e.option_name}' (ID: {e.option_id})")
    print(f"Validation type: {e.validation_type}")
    print(f"Failed at step {e.failing_step_index} with action {e.action_repr}")
    print(f"Environment state: {e.short_obs_summary}")
    if e.base_error:
        print(f"Underlying error: {e.base_error}")
```

### SMDPOptionExecutionError

Raised during **option execution** when a primitive action fails at runtime.

```python
from smdpfier.errors import SMDPOptionExecutionError

try:
    obs, reward, term, trunc, info = env.step(problematic_option_index)
except SMDPOptionExecutionError as e:
    print(f"Execution failed for option '{e.option_name}' (ID: {e.option_id})")
    print(f"Failed at step {e.failing_step_index} with action {e.action_repr}")
    print(f"Environment state: {e.short_obs_summary}")
    if e.base_error:
        print(f"Underlying error: {e.base_error}")
```

## Error Context Fields

Both error types provide comprehensive context information:

| Field | Type | Description |
|-------|------|-------------|
| `option_name` | `str` | Human-readable name of the failing option |
| `option_id` | `str` | Unique identifier for the option (hash-based) |
| `failing_step_index` | `int` | Which action in the sequence failed (0-indexed) |
| `action_repr` | `str` | String representation of the failing action |
| `short_obs_summary` | `str` | Abbreviated environment state description |
| `base_error` | `Exception \| None` | Original exception that caused the failure |
| `validation_type` | `str` | Type of validation that failed (ValidationError only) |

### Error Context Example

```python
# When this option fails:
problematic_option = Option([0, 1, 99], "problematic-sequence")

# You get detailed context:
# SMDPOptionExecutionError: Option execution failed
#   option_name: "problematic-sequence"
#   option_id: "abc123def456..."
#   failing_step_index: 2
#   action_repr: "99"
#   short_obs_summary: "obs=[0.1, -0.3, 0.05, ...] (CartPole-v1)"
#   base_error: ValueError("Invalid action 99 for Discrete(2) action space")
```

## Common Error Scenarios

### Invalid Action for Action Space

**Most common error** - action not in environment's action space.

```python
# CartPole has Discrete(2) action space (actions 0, 1)
invalid_option = Option([0, 1, 2], "invalid-action")  # Action 2 is invalid

try:
    obs, reward, term, trunc, info = env.step(invalid_option)
except SMDPOptionExecutionError as e:
    print(f"Invalid action {e.action_repr} at step {e.failing_step_index}")
    # Output: "Invalid action 2 at step 2"
```

**Solution strategies:**
```python
# Strategy 1: Validate options at creation
def create_safe_option(actions, name, action_space):
    """Create option with action space validation."""
    if isinstance(action_space, gym.spaces.Discrete):
        valid_actions = list(range(action_space.n))
        invalid_actions = [a for a in actions if a not in valid_actions]
        if invalid_actions:
            raise ValueError(f"Invalid actions {invalid_actions} for {action_space}")
    
    return Option(actions, name)

# Strategy 2: Use masking to prevent invalid options
def safe_availability(obs):
    """Only allow options with valid actions."""
    return [0, 1, 2]  # Indices of options with valid actions only
```

### Continuous Action Shape Mismatch

**Continuous environments** require actions with correct shape.

```python
# Pendulum expects actions with shape (1,)
wrong_shape_option = Option([[1.0, 2.0]], "wrong-shape")  # Shape (2,) - wrong!
correct_option = Option([[1.5]], "correct-shape")         # Shape (1,) - correct

try:
    obs, reward, term, trunc, info = env.step(wrong_shape_option)
except SMDPOptionExecutionError as e:
    print(f"Shape error: {e.base_error}")
    # Output: "Shape error: ValueError('Expected action shape (1,), got (2,)')"
```

### Environment State Issues

Some environments have **state-dependent action validity**.

```python
# Taxi environment - pickup only valid at passenger locations
def taxi_example():
    env = SMDPfier(gym.make("Taxi-v3"), ...)
    
    # This might fail if taxi is not at passenger location
    pickup_option = Option([4], "pickup")  # Action 4 = pickup
    
    try:
        obs, reward, term, trunc, info = env.step(pickup_option)
    except SMDPOptionExecutionError as e:
        if "pickup" in str(e.base_error).lower():
            print("Pickup failed - taxi not at passenger location")
            # Try movement instead
            move_option = Option([0], "move-south")
            obs, reward, term, trunc, info = env.step(move_option)
```

### Early Termination During Option

**Episode termination** during option execution is handled gracefully (not an error), but understanding the behavior is important.

```python
# Long option that might cause episode termination
long_option = Option([0] * 10, "ten-left-actions")

obs, reward, term, trunc, info = env.step(long_option)

# Check if option completed
smdp_info = info["smdp"]
if smdp_info["terminated_early"]:
    print(f"Episode ended after {smdp_info['k_exec']} of {smdp_info['option']['len']} actions")
    print(f"Partial duration: {smdp_info['duration_exec']} ticks")
```

## Error Handling Patterns

### Basic Try-Catch Pattern

```python
def robust_step(env, action):
    """Robust step with error handling."""
    try:
        return env.step(action)
    except SMDPOptionValidationError as e:
        print(f"Precheck failed: {e.option_name} at step {e.failing_step_index}")
        # Return safe fallback or re-raise
        raise
    except SMDPOptionExecutionError as e:
        print(f"Execution failed: {e.option_name} with action {e.action_repr}")
        # Return safe fallback or re-raise  
        raise
```

### Fallback Strategy Pattern

```python
def execute_with_fallback(env, primary_option, fallback_options):
    """Try primary option, fall back to alternatives on failure."""
    
    # Try primary option
    try:
        return env.step(primary_option)
    except (SMDPOptionValidationError, SMDPOptionExecutionError) as e:
        print(f"Primary option failed: {e}")
        
        # Try fallback options
        for i, fallback in enumerate(fallback_options):
            try:
                print(f"Trying fallback {i+1}: {fallback.name}")
                return env.step(fallback)
            except (SMDPOptionValidationError, SMDPOptionExecutionError):
                continue
        
        # All options failed
        raise RuntimeError("All options failed, no valid actions available")

# Usage
primary = Option([0, 1, 0, 1], "complex-sequence")
fallbacks = [
    Option([0, 1], "simple-sequence"),
    Option([0], "minimal-action"),
]

obs, reward, term, trunc, info = execute_with_fallback(env, primary, fallbacks)
```

### Logging and Monitoring Pattern

```python
import logging

logger = logging.getLogger("smdp_errors")

def monitored_step(env, action):
    """Step with comprehensive error logging."""
    try:
        result = env.step(action)
        
        # Log successful execution details
        smdp_info = result[4]["smdp"]  # info["smdp"]
        logger.info(f"Option '{smdp_info['option']['name']}' executed successfully")
        logger.debug(f"Steps: {smdp_info['k_exec']}, Duration: {smdp_info['duration_exec']}")
        
        return result
        
    except SMDPOptionValidationError as e:
        logger.error(f"Validation error: {e.option_name} failed at step {e.failing_step_index}")
        logger.error(f"Action: {e.action_repr}, State: {e.short_obs_summary}")
        logger.debug(f"Full error: {e}")
        raise
        
    except SMDPOptionExecutionError as e:
        logger.error(f"Execution error: {e.option_name} failed at step {e.failing_step_index}")
        logger.error(f"Action: {e.action_repr}, Base error: {e.base_error}")
        logger.debug(f"Full error: {e}")
        raise
```

### Error Recovery for RL Training

```python
class ErrorResilientAgent:
    def __init__(self, env):
        self.env = env
        self.error_counts = {}
        self.max_error_threshold = 5
    
    def step_with_recovery(self, action):
        """RL training step with error recovery."""
        try:
            return self.env.step(action)
            
        except (SMDPOptionValidationError, SMDPOptionExecutionError) as e:
            # Track error frequency
            error_key = (e.option_name, e.failing_step_index)
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # If this error happens too often, it's a systematic issue
            if self.error_counts[error_key] > self.max_error_threshold:
                print(f"Systematic error detected: {error_key}")
                # Could trigger option blacklisting, environment reset, etc.
            
            # For training, reset environment and continue
            print(f"Error during training: {e.option_name}, resetting episode")
            obs, info = self.env.reset()
            return obs, 0.0, True, False, info  # Return terminal step
    
    def get_error_statistics(self):
        """Get error frequency statistics."""
        return dict(self.error_counts)
```

## Debugging Techniques

### Option Validation

```python
def validate_option_thoroughly(env, option):
    """Comprehensive option validation."""
    print(f"Validating option: {option.name}")
    print(f"Actions: {option.actions}")
    print(f"Action count: {len(option.actions)}")
    
    # Check action space compatibility
    action_space = env.unwrapped.action_space
    print(f"Environment action space: {action_space}")
    
    for i, action in enumerate(option.actions):
        if isinstance(action_space, gym.spaces.Discrete):
            if not (0 <= action < action_space.n):
                print(f"  ❌ Action {i}: {action} invalid for {action_space}")
            else:
                print(f"  ✅ Action {i}: {action} valid")
        elif isinstance(action_space, gym.spaces.Box):
            action = np.array(action)
            if action.shape != action_space.shape:
                print(f"  ❌ Action {i}: shape {action.shape} != {action_space.shape}")
            elif not action_space.contains(action):
                print(f"  ❌ Action {i}: {action} outside bounds {action_space}")
            else:
                print(f"  ✅ Action {i}: {action} valid")
```

### Error Pattern Analysis

```python
def analyze_error_patterns(error_log):
    """Analyze common error patterns from logs."""
    error_types = {}
    failing_actions = {}
    failing_options = {}
    
    for error in error_log:
        # Group by error type
        error_type = type(error).__name__
        error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Group by failing action
        action = error.action_repr
        failing_actions[action] = failing_actions.get(action, 0) + 1
        
        # Group by option name
        option = error.option_name
        failing_options[option] = failing_options.get(option, 0) + 1
    
    print("Error Analysis:")
    print(f"Error types: {error_types}")
    print(f"Most failing actions: {sorted(failing_actions.items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"Most failing options: {sorted(failing_options.items(), key=lambda x: x[1], reverse=True)[:5]}")
```

### Interactive Debugging

```python
def debug_option_interactively(env, option):
    """Step through option execution interactively."""
    print(f"Debug mode for option: {option.name}")
    print(f"Actions to execute: {option.actions}")
    
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    
    for i, action in enumerate(option.actions):
        print(f"\nStep {i}: About to execute action {action}")
        input("Press Enter to continue...")
        
        try:
            obs, reward, term, trunc, info = env.step(action)
            print(f"Result: obs={obs[:3]}..., reward={reward}, term={term}, trunc={trunc}")
            
            if term or trunc:
                print("Episode terminated!")
                break
                
        except Exception as e:
            print(f"Error at step {i}: {e}")
            break
```

## Error Prevention

### Option Design Guidelines

```python
# ✅ Good: Simple, robust options
good_options = [
    Option([0], "single-left"),           # Minimal failure risk
    Option([1], "single-right"),          # Minimal failure risk  
    Option([0, 1], "balanced"),           # Self-correcting
]

# ❌ Avoid: Complex, risky options
risky_options = [
    Option([0] * 20, "extreme-long"),     # High termination risk
    Option([999], "invalid-action"),      # Invalid action
    Option([0, 1, 0, 1, 0, 1], "too-long"), # Long sequence risk
]
```

### Environment-Specific Safety

```python
def create_safe_cartpole_options():
    """Create CartPole options with built-in safety."""
    # CartPole actions: 0 (left), 1 (right)
    safe_options = [
        Option([0], "left"),
        Option([1], "right"),
        Option([0, 1], "left-right"),     # Balanced
        Option([1, 0], "right-left"),     # Balanced
        # Avoid long sequences that might cause termination
    ]
    return safe_options

def create_safe_pendulum_options():
    """Create Pendulum options with proper action ranges."""
    # Pendulum action space: Box([-2.0], [2.0])
    safe_options = [
        Option([[-2.0]], "max-left"),
        Option([[2.0]], "max-right"),
        Option([[0.0]], "no-torque"),
        Option([[-1.0], [1.0]], "swing"),
        Option([[0.5], [0.5]], "gentle-right"),
    ]
    return safe_options
```

## Summary

| Error Type | When It Occurs | How to Handle |
|------------|----------------|---------------|
| `SMDPOptionValidationError` | Precheck failure | Disable precheck or fix option |
| `SMDPOptionExecutionError` | Runtime action failure | Use fallbacks or fix option |
| Early termination | Episode ends during option | Normal behavior, check `terminated_early` |

**Best Practices:**
1. **Design robust options** with minimal failure risk
2. **Use action masking** to prevent invalid options
3. **Implement fallback strategies** for critical applications
4. **Log errors comprehensively** for debugging
5. **Test options thoroughly** in development

**Key Takeaway**: SMDPfier's detailed error context makes debugging much easier than generic exceptions. Use this information to build robust, error-resilient SMDP applications.

---

**Next**: [FAQ](faq.md) | **See Also**: [Masking and Precheck](masking_and_precheck.md)
