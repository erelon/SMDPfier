# Releasing SMDPfier

This document outlines the complete process for releasing a new version of SMDPfier.

## Prerequisites

Before releasing, ensure you have:

1. **Development Environment Setup**
   ```bash
   pip install -e .[dev]
   ```

2. **Repository Access**
   - Push access to the main repository
   - Maintainer access for creating releases

3. **Clean Working Directory**
   ```bash
   git status  # Should show no uncommitted changes
   ```

## Pre-Release Checklist

### 1. Code Quality Checks
Run the full test suite locally:

```bash
# Lint check
ruff check smdpfier tests examples
ruff format --check smdpfier tests examples

# Type checking
mypy smdpfier

# Run tests
pytest tests/ -v --cov=smdpfier

# Test examples run without errors
python examples/cartpole_index_static.py
python examples/pendulum_direct_continuous.py
python examples/taxi_index_dynamic_mask.py
```

### 2. Documentation Verification
```bash
# Build documentation
mkdocs build --strict

# Serve locally to review
mkdocs serve
```

### 3. Version Update
Update the version in `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"  # Update to new version
```

### 4. Changelog Update
Update `CHANGELOG.md`:

1. Move items from `[Unreleased]` to a new version section
2. Add release date
3. Create new empty `[Unreleased]` section

Example:
```markdown
## [Unreleased]

## [0.1.1] - 2025-11-12
### Added
- New feature X
- Enhancement Y

### Fixed
- Bug fix Z
```

## Release Process

### Step 1: Prepare Release Branch
```bash
# Create and switch to release branch
git checkout -b release/v0.1.1
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.1"
git push origin release/v0.1.1
```

### Step 2: Create Pull Request
1. Open PR from `release/v0.1.1` to `main`
2. Title: `Release v0.1.1`
3. Include changelog summary in PR description
4. Wait for CI to pass
5. Get approval from maintainers

### Step 3: Merge and Tag
```bash
# Switch to main and pull the merged changes
git checkout main
git pull origin main

# Create and push the release tag
git tag v0.1.1
git push origin v0.1.1
```

### Step 4: Verify Build
The release workflow will automatically:
1. Build wheel and source distribution
2. Run `twine check` on artifacts  
3. Test installation on multiple Python versions
4. Upload build artifacts

Monitor the GitHub Actions workflow at:
`https://github.com/your-org/smdpfier/actions`

### Step 5: Create GitHub Release
1. Go to GitHub Releases page
2. Click "Create a new release"
3. Select the tag `v0.1.1`
4. Title: `SMDPfier v0.1.1`
5. Copy changelog entries for this version
6. Attach the built artifacts from the workflow
7. Publish the release

## Post-Release Verification

### 1. Test Installation from Built Artifacts
Download the wheel from the GitHub release and test:

```bash
# Create clean environment
python -m venv test-env
source test-env/bin/activate

# Install from wheel
pip install path/to/smdpfier-X.Y.Z-py3-none-any.whl

# Verify installation
python -c "import smdpfier; print(f'SMDPfier v{smdpfier.__version__} installed')"
python -c "from smdpfier import SMDPfier, Option; print('Core imports work')"

# Test a simple example
python -c "
from smdpfier import SMDPfier, Option
from smdpfier.defaults import ConstantOptionDuration
import gymnasium as gym

env = gym.make('CartPole-v1')
options = [Option('left', [0]), Option('right', [1])]
wrapped = SMDPfier(env, options_provider=options, duration_fn=ConstantOptionDuration(5))
print('SMDPfier wrapper created successfully')
"

deactivate
rm -rf test-env
```

### 2. Update Documentation
If using external documentation hosting:
1. Trigger documentation rebuild
2. Verify new version appears in docs
3. Check that API references are current

## Emergency Procedures

### Yanking a Release
If a critical issue is discovered:

1. **GitHub Release**: Edit the release and mark as "pre-release"
2. **Document Issue**: Create issue tracking the problem
3. **Hot Fix**: Prepare patch release with fix

### Rolling Back
```bash
# Delete the problematic tag
git tag -d v0.1.1
git push origin :refs/tags/v0.1.1

# Revert version in main if needed
git revert <commit-hash>
git push origin main
```

## PyPI Publishing (Future)

When ready to publish to PyPI, update the release workflow to include:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  if: startsWith(github.ref, 'refs/tags/')
  with:
    user: __token__
    password: ${{ secrets.PYPI_API_TOKEN }}
```

And add the PyPI API token to repository secrets.

## Versioning Strategy

SMDPfier follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New functionality, backwards compatible
- **PATCH** (0.0.X): Bug fixes, backwards compatible

### Version Bumping Guidelines

- **Patch (0.1.1)**: Bug fixes, documentation updates, minor improvements
- **Minor (0.2.0)**: New features, new defaults, new optional parameters
- **Major (1.0.0)**: Breaking API changes, removed functionality

## Troubleshooting

### Build Failures
```bash
# Clean build artifacts
rm -rf dist/ build/ *.egg-info/

# Reinstall build dependencies
pip install --upgrade build twine

# Rebuild
python -m build
```

### CI/CD Issues
- Check GitHub Actions logs for specific errors
- Verify all required secrets are configured
- Ensure branch protection rules allow the release process

### Installation Issues
- Verify Python version compatibility (>=3.9)
- Check for missing dependencies
- Test in clean virtual environment

## Contacts

For release-related questions:
- Primary maintainer: [Your Name] ([email])
- Backup maintainer: [Backup Name] ([email])
- Issues: https://github.com/your-org/smdpfier/issues
