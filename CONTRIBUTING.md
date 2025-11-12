# Contributing to SMDPfier

Thank you for your interest in contributing to SMDPfier! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Clone and Install
```bash
git clone https://github.com/your-org/smdpfier.git
cd smdpfier
pip install -e .[dev]
```

### 2. Run Tests and Quality Checks
```bash
# Run all tests
pytest tests/ -v

# Code quality checks
ruff check smdpfier tests examples
ruff format --check smdpfier tests examples
mypy smdpfier

# Test examples
python examples/cartpole_index_static.py
python examples/taxi_index_dynamic_mask.py
python examples/pendulum_direct_continuous.py
```

### 3. Build and Validate Package
```bash
python -m build
twine check dist/*
```

## Types of Contributions

### 🐛 Bug Reports
- Use the issue template (when available)
- Include minimal reproduction case
- Specify Python version and dependencies
- Include error messages and stack traces

### 🚀 Feature Requests
- Describe the use case and motivation
- Consider if it fits SMDPfier's scope (SMDP behavior for Gymnasium)
- Propose API design if applicable

### 📝 Documentation Improvements
- Fix typos, improve clarity, add examples
- Ensure consistency with existing style
- Test documentation builds: `mkdocs serve`

### 💻 Code Contributions
- Follow existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Ensure all quality checks pass

## Coding Standards

### Code Style
- Follow PEP 8 and existing project patterns
- Use type hints throughout
- Keep functions focused and well-documented
- Use descriptive variable names

### Testing
- Write tests for new functionality
- Maintain or improve test coverage
- Test edge cases and error conditions
- Use descriptive test names

### Documentation
- Update docstrings for new/changed functions
- Include examples in docstrings when helpful
- Update relevant documentation files
- Ensure cross-references are accurate

## Pull Request Process

### 1. Before Starting
- Check existing issues and PRs to avoid duplication
- Discuss major changes in an issue first
- Fork the repository and create a feature branch

### 2. Development
- Make focused changes (one feature/fix per PR)
- Write clear commit messages
- Keep commits atomic and logical
- Test your changes thoroughly

### 3. Submitting
- Fill out the PR template (when available)
- Ensure all CI checks pass
- Request review from maintainers
- Address feedback promptly

### 4. After Approval
- Squash commits if requested
- Maintainers will handle merging

## Release Process

Releases are handled by maintainers following the process documented in `RELEASING.md`. Contributors don't need to worry about versioning or releases.

## Code of Conduct

This project follows a standard code of conduct. Be respectful, inclusive, and constructive in all interactions.

## Getting Help

- Check the [FAQ](docs/faq.md) for common questions
- Review existing [documentation](docs/)
- Open an issue for bugs or feature requests
- Join discussions in issues and PRs

## Recognition

All contributors are recognized in releases and the project history. Thank you for helping make SMDPfier better!

## Development Tips

### Understanding the Codebase
- Start with `smdpfier/wrapper.py` for the main logic
- Check `smdpfier/option.py` for the Option class
- Look at `examples/` for usage patterns
- Read `tests/` for expected behavior

### Common Development Tasks

#### Adding a New Duration Function
1. Add to `smdpfier/defaults/durations.py`
2. Follow existing patterns (ConstantOptionDuration, etc.)
3. Add tests to `tests/test_durations.py`
4. Update documentation in `docs/api.md`

#### Adding a New Reward Aggregator
1. Add to `smdpfier/defaults/rewards.py`
2. Follow signature: `Callable[[list[float]], float]`
3. Add tests to appropriate test file
4. Document in API reference

#### Fixing a Bug
1. Write a test that reproduces the bug
2. Fix the issue while keeping the test passing
3. Ensure no regressions in existing tests
4. Update documentation if needed

### Testing Strategy
- Unit tests for individual components
- Integration tests for full workflows
- Example tests to ensure they run correctly
- Edge case testing for error conditions

Thank you for contributing to SMDPfier!
