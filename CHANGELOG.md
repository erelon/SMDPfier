# Changelog

All notable changes to SMDPfier will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-11-19

### Changed - **BREAKING: Simplified Time Semantics**
- **Duration = k_exec**: Duration now simply equals the number of primitive actions executed. Each primitive action = 1 tick.
- **Removed `duration_fn` parameter**: No longer needed as duration is automatically k_exec
- **Removed `partial_duration_policy` parameter**: No longer needed with simplified duration model  
- **Removed `time_units` attribute**: Always implicitly "ticks" (= primitive actions)
- **Default reward = sum**: Macro reward now defaults to sum of per-step rewards (`sum_rewards`)
- **Simplified info structure**: `info["smdp"]["duration"]` now contains just k_exec (removed `duration_planned` and `duration_exec`)
- **Removed `reward_rate`**: Rate-based rewards removed from defaults; users can implement custom aggregators if needed

### Removed - **BREAKING: Duration Machinery**
- **`smdpfier.defaults.durations` module**: Completely removed (ConstantOptionDuration, RandomOptionDuration, ConstantActionDuration, RandomActionDuration, MapActionDuration)
- **Duration validation utilities**: Removed `validate_duration_function_output` from utils

### Added
- **New test coverage**: Added `test_duration_equals_kexec` to verify duration always equals k_exec

### Migration Guide
Users upgrading from 0.1.x need to:
1. Remove `duration_fn` parameter from SMDPfier initialization
2. Remove `partial_duration_policy` parameter if used
3. Remove imports from `smdpfier.defaults.durations`
4. Access duration via `info["smdp"]["duration"]` (equals k_exec)
5. Explicitly pass `reward_agg=sum_rewards` if using default (or omit for same behavior)
6. If using rate-based rewards, implement custom aggregator: `lambda rewards: sum(rewards) / len(rewards)` or similar

## [0.1.0] - Previous Release

### Added - Rate-Based Default Reward (ENV-side Duration Awareness)
- **Rate-based reward as default**: Macro reward now defaults to `sum(rewards) / max(1, duration_exec)`, making duration environmentally significant
- **`reward_rate` function**: New default reward aggregator that computes reward per unit time
- **Flexible reward_agg signatures**: Support for both simple `(rewards)` and advanced `(rewards, duration_exec, per_action_durations)` signatures with automatic detection
- **Comprehensive rate tests**: Tests verifying rate calculation, duration sensitivity, and backward compatibility
- **Zero-duration edge case handling**: Denominator of `max(1, duration_exec)` prevents division by zero

### Changed - Rate-Based Default Reward
- **Default reward aggregator**: Changed from `sum_rewards` to rate-based (sum/duration)
- **`sum_rewards` marked as legacy**: Now documented as deprecated in favor of `reward_rate`
- **Reward aggregation logic**: Enhanced to detect function signatures and support flexible parameter passing
- **Documentation**: Updated to present rate as the recommended SMDP reward objective

### Deprecated - Rate-Based Default Reward
- **`sum_rewards`**: Still available for backward compatibility but marked as legacy. Users wanting sum behavior must explicitly pass `reward_agg=sum_rewards`

### Added - Agent 4b Test Enablement & Repo Cleanup
- **Complete test suite** with all pytest.skip() calls replaced with real test implementations
- **Duration function tests** covering ConstantOptionDuration, RandomOptionDuration, ConstantActionDuration, RandomActionDuration, and MapActionDuration
- **Error handling tests** for SMDPOptionValidationError and SMDPOptionExecutionError with runtime failure scenarios
- **Partial duration policy tests** for proportional, full, and zero policies with early termination scenarios
- **SMDPfier integration tests** verifying duration metadata propagation and option execution
- **CI workflow** with GitHub Actions for automated testing across Python 3.9-3.12
- **Test coverage** for all major SMDPfier functionality including action masking and SMDP info payloads

### Changed - Agent 4b Test Enablement & Repo Cleanup
- **Repository cleanup** removed all __pycache__ directories and *.pyc files from version control
- **Scaffolding removal** deleted validate_scaffolding.py helper file
- **Test structure** enhanced test_masks_and_precheck.py with smoke test functionality

### Added - Agent 6 Packaging, CI & Release Prep
- **Modern build system** using hatchling instead of setuptools for better packaging
- **Complete CI/CD pipeline** with GitHub Actions for testing, building, and release preparation
- **Multi-Python version testing** (3.9-3.12) in CI with ruff, mypy, and pytest
- **Automated build and distribution validation** with twine checks on tag pushes
- **Release workflow** that builds wheels and source distributions automatically
- **Development dependencies** including build tools (build, twine) in dev extras
- **RELEASING.md** with comprehensive step-by-step release instructions
- **Code quality improvements** with automatic linting fixes and error handling improvements

### Changed - Agent 6 Packaging, CI & Release Prep
- **Build backend** migrated from setuptools to hatchling for modern packaging
- **Type annotations** updated to use modern syntax (X | Y instead of Union[X, Y])
- **Exception handling** improved with proper exception chaining using 'from' syntax
- **Import organization** standardized across all modules
- **Code style** cleaned up to pass all ruff and mypy checks

### Fixed - Agent 6 Packaging, CI & Release Prep
- **Duplicate class definitions** in defaults modules that caused import errors
- **Orphaned function references** that broke functionality
- **Boolean comparisons** in tests using proper truthiness checks
- **Loop variables** renamed to avoid linting warnings
- **Test issues** with reward aggregation function usage

### Infrastructure - Agent 6 Packaging, CI & Release Prep
- **CI pipeline** ready for automated testing on pull requests and releases
- **Build process** validated with clean installs and twine checks
- **Release preparation** complete with documentation and automated workflows
- **Package structure** optimized for PyPI distribution (ready but not published)

### Added - Agent 5 Documentation Polish
- **Comprehensive README.md** with badges, quickstart, and feature highlights
- **Complete documentation overhaul** with consistent cross-linking and navigation
- **Crystal-clear duration vs steps distinction** throughout all documentation
- **Extensive SMDP discounting examples** with γ^{ticks} formulations
- **Detailed API reference** with configuration patterns and built-in defaults
- **Comprehensive FAQ** addressing common misconceptions and troubleshooting
- **Advanced masking and precheck guide** with practical examples
- **Error handling documentation** with debugging techniques and recovery patterns
- **Interface comparison guide** (index vs direct) with selection criteria
- **Duration guide** explaining ticks, partial policies, and SMDP discounting
- **Performance optimization tips** and best practices throughout docs
- **Visual organization** with tables, emojis, and consistent formatting

### Changed - Agent 5 Documentation Polish  
- **Enhanced index.md** with improved navigation and core concepts explanation
- **Restructured all documentation** for better user journey flow
- **Standardized terminology** (ticks, steps, options, interfaces) across all docs
- **Improved code examples** with working implementations and clear explanations
- **Enhanced cross-referencing** between documentation sections

### Documentation Highlights
- **Duration vs Steps**: Made this critical distinction unmistakable throughout docs
- **SMDP Discounting**: Explicit γ^{ticks} vs γ^{steps} comparisons with practical examples
- **Interface Selection**: Clear guidance on when to use index vs direct interfaces
- **Error Handling**: Comprehensive debugging and recovery strategies
- **User Journey**: Documentation structured for different user types (new users, RL practitioners, researchers)

### Developer Experience
- **Complete API documentation** with all parameters and examples
- **Built-in defaults documentation** with usage patterns
- **Error context explanations** for easier debugging
- **Performance and optimization guidance**
- **Extensive troubleshooting section**

## [0.1.0] - TBD

### Added
- Initial SMDPfier implementation
- Option class for action sequences
- Duration functions for temporal metadata
- Index and direct action interfaces
- Action masking and precheck validation
- Error handling with detailed context
- Built-in default generators and functions
- Comprehensive test suite
- Example implementations

### Technical Features
- Semi-Markov Decision Process wrapper for Gymnasium environments
- Duration metadata for proper SMDP discounting
- Static and dynamic option support
- Discrete and continuous action space compatibility
- Partial duration policies for early termination
- Overflow handling for dynamic options
- Rich execution metadata in info payload

---

## Release Notes Template

### [Version] - YYYY-MM-DD

#### Added
- New features and capabilities

#### Changed  
- Changes to existing functionality

#### Deprecated
- Features marked for removal

#### Removed
- Removed features

#### Fixed
- Bug fixes

#### Security
- Security-related changes
