# PRIMER.md - SMDPfier Current State (Post Rate-Based Reward)

## Project Overview

SMDPfier is a **fully tested** Gymnasium wrapper that transforms any environment into a Semi-Markov Decision Process (SMDP) by enabling **Options** (action sequences) with **duration metadata** for proper temporal discounting.

**Core Innovation**: Separates execution (env steps) from time (ticks for discounting), and uses **rate-based rewards by default** to make duration environmentally significant.

**Testing Status**: Complete test suite with all pytest.skip() calls replaced by real implementations. CI/CD pipeline established with GitHub Actions for automated testing across Python 3.9-3.12.

**Version**: 0.1.0+ (rate-based reward default)

## Architecture Summary

### Core Classes
- **SMDPfier**: Main wrapper class extending gym.Wrapper
- **Option**: Represents action sequences with metadata
- **Built-in Defaults**: Ready-to-use generators and functions in `smdpfier.defaults.*`

### Key Components
- **Action Interfaces**: Index (discrete) vs Direct (intuitive)
- **Duration Functions**: Scalar (per-option) vs List (per-action)
- **Option Providers**: Static lists or dynamic generators
- **Error Handling**: Specialized exceptions with rich context
- **Action Masking**: State-dependent option availability
- **Rate-Based Rewards**: Default macro reward = sum(rewards) / max(1, duration_exec)

## Critical Concepts

### 1. Steps vs Duration (Most Important!)
- **Steps**: Number of `env.step()` calls (determined by option length)
- **Duration**: Abstract "ticks" for SMDP discounting AND reward rate calculation
- **KEY**: Duration affects macro reward (via rate) and SMDP discounting (γ^{ticks})

### 2. Rate-Based Reward (Default as of 0.1.0)
- **Formula**: `macro_reward = sum(primitive_rewards) / max(1, duration_exec)`
- **Purpose**: Makes duration environmentally significant, encourages time efficiency
- **Example**: Same total reward (4.0) with duration 10 → reward 0.4, with duration 20 → reward 0.2
- **Override**: Pass `reward_agg=sum_rewards` for legacy sum behavior

### 3. SMDP Discounting
- **Standard MDP**: γ^{steps}
- **SMDP**: γ^{ticks} using `info["smdp"]["duration_exec"]`
- **Cumulative**: Track total elapsed time across options

### 4. Action Interfaces
- **Index**: `Discrete(max_options)` action space for RL training
- **Direct**: Pass Option objects directly for scripting/testing

## API Structure

```python
SMDPfier(
    env,                                    # Base Gymnasium environment
    options_provider,                       # Static list or dynamic callable  
    duration_fn,                           # Returns int or list[int]
    reward_agg=None,                       # None=rate, or custom aggregator
    action_interface="index",              # "index" or "direct"
    max_options=None,                      # Required for index interface
    availability_fn=None,                  # Action masking function
    precheck=False,                        # Validate before execution
    partial_duration_policy="proportional", # Handle early termination
    time_units="ticks",                    # Metadata only
    rng_seed=None                          # Reproducibility
)
```

## Info Payload Structure

Every step returns rich metadata in `info["smdp"]`:

```python
{
    "option": {"id": "...", "name": "...", "len": 3, "meta": {}},
    "k_exec": 3,                    # Steps actually executed
    "rewards": [1.0, 1.0, 1.0],    # Per-step rewards
    "duration_planned": 10,         # Expected ticks
    "duration_exec": 10,           # Actual ticks (early termination aware)
    "terminated_early": False,      # Episode ended during option?
    "time_units": "ticks",         # Always "ticks"
    "action_mask": [1, 1, 0],      # Available options (index interface)
    "num_dropped": 0               # Overflow options dropped (index interface)
}
```

## Built-in Defaults Location

### Option Generators (`smdpfier.defaults.options`)
- `RandomStaticLen(length, action_space_size, num_options)`
- `RandomVarLen(min_length, max_length, action_space_size, num_options)`

### Duration Functions (`smdpfier.defaults.durations`)
- `ConstantOptionDuration(duration)` - Scalar
- `RandomOptionDuration(min_duration, max_duration)` - Scalar  
- `ConstantActionDuration(duration)` - List
- `RandomActionDuration(min_duration, max_duration)` - List
- `MapActionDuration(action_map, default_duration)` - List

### Reward Aggregation (`smdpfier.defaults.rewards`)
- `reward_rate` - **Default**: sum(rewards) / max(1, duration_exec)
- `sum_rewards` - **Legacy**: Sum per-step rewards (ignores duration)
- `mean_rewards` - Average per-step rewards
- `discounted_sum(gamma)` - Discount per-step rewards

## Error Handling

### Exception Types
- `SMDPOptionValidationError` - Precheck failures
- `SMDPOptionExecutionError` - Runtime failures

### Error Context Fields
- `option_name`, `option_id` - Option identification
- `failing_step_index` - Which action failed (0-indexed)
- `action_repr` - String representation of failing action
- `short_obs_summary` - Environment state summary
- `base_error` - Original exception

## Current Implementation Status

**✅ Implemented**: Core SMDPfier, Option class, duration functions, interfaces, masking, error handling, built-in defaults  
**✅ Documentation**: Complete - README, API reference, duration guide, interfaces, masking, errors, FAQ  
**✅ Testing**: Comprehensive test suite with all pytest.skip() calls replaced by real implementations  
**✅ Test Coverage**: Duration functions, error handling, partial policies, SMDPfier integration, smoke tests  
**✅ CI/CD**: GitHub Actions workflows for automated testing across Python 3.9-3.12  
**✅ Repository**: Clean state with all build artifacts and scaffolding files removed  
**✅ Packaging**: Modern hatchling build system, proper pyproject.toml configuration, dev dependencies  
**✅ Code Quality**: All linting issues resolved, modern type annotations, proper exception handling

## File Structure

Key files: README.md, docs/*.md (comprehensive documentation), smdpfier/ (implementation), examples/ (working code), tests/ (validation).

## Key Design Decisions

1. **Duration as metadata only** - never controls execution, prevents confusion
2. **Stable Option IDs** - hash-based for consistent identification
3. **Rich info payload** - comprehensive execution metadata
4. **Interface separation** - index for RL, direct for scripting
5. **Graceful error handling** - detailed context for debugging

## Usage Patterns

**RL Training**: Use index interface with `info["smdp"]["duration_exec"]` for γ^{ticks} discounting  
**Direct Control**: Use direct interface passing Option objects  
**Dynamic Options**: Use generator functions with max_options parameter

## Infrastructure Status (Agent 6 Complete)

**✅ Build System**: Hatchling-based packaging, clean builds, twine validation passes  
**✅ CI/CD Pipeline**: Multi-Python testing, automated releases on tags, artifact management  
**✅ Code Quality**: All ruff/mypy issues resolved, modern Python practices throughout  
**✅ Release Automation**: Tag-based releases with build validation and installation testing  
**🔧 Ready for PyPI**: Publishing workflow prepared (requires API token setup)

## Next Steps for Future Agents

**Agent 5 Priority - Documentation Polish**:
- Enhance documentation with visual examples and clearer explanations
- Add comprehensive usage guides and troubleshooting sections
- Create detailed API documentation with examples
- Polish README and documentation consistency

**Agent 6 Priority - Packaging & CI**:
- Set up complete CI/CD pipeline with automated releases
- Configure modern Python packaging with hatchling
- Add comprehensive linting, type checking, and code quality tools
- Prepare for PyPI publishing with proper build validation

**Future Enhancements**: 
- Performance optimization and benchmarking
- Extended testing (integration, property-based, performance)
- Community features (issue templates, contribution guidelines)
- RL framework integration guides

## Success Metrics

**Core Implementation**: ✅ Complete and validated with comprehensive test suite  
**Test Enablement (Agent 4b)**: ✅ Complete - all tests implemented and running, CI established  
**Documentation (Agent 5)**: ⏳ Planned - crystal clear concepts, comprehensive API, troubleshooting  
**Packaging & CI (Agent 6)**: ⏳ Planned - production-ready infrastructure, automated quality assurance  

---

**Current Status**: **Fully tested and CI-enabled**. Core implementation complete with comprehensive test coverage. Repository is clean and ready for documentation polish (Agent 5) and packaging (Agent 6).
