# Contributing to SC2-LSTM-Bot

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to learn and build something cool together.

## Getting Started

### Prerequisites

1. Python 3.8 or higher
2. StarCraft II installed
3. NVIDIA GPU with CUDA support (for training)
4. Git

### Finding Issues to Work On

- Look for issues labeled `good first issue` for beginner-friendly tasks
- Issues labeled `help wanted` are actively seeking contributors
- Feel free to open new issues for bugs or feature requests

## Development Setup

1. **Fork the repository**

2. **Clone your fork**
```bash
git clone https://github.com/YOUR_USERNAME/sc2-lstm-bot.git
cd sc2-lstm-bot
```

3. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

5. **Set up pre-commit hooks** (optional but recommended)
```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-zerg-support`
- `fix/coordinate-transform-bug`
- `docs/update-readme`
- `refactor/simplify-reward-system`

### Commit Messages

Follow conventional commits format:

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `perf`: Performance improvement
- `chore`: Maintenance tasks

### Example:
```
feat(combat): add retreat_to_base macro

Implements a new combat macro that retreats all army units to the
nearest Command Center when called.

- Added retreat_to_base function in terran_actions.py
- Added pre_retreat_to_base precondition
- Updated ACTION_REGISTRY and ACTION_TO_INDEX

Fixes #42
```

## Pull Request Process

1. **Update your fork**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Create a branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make your changes and commit**
```bash
git add .
git commit -m "feat: add your feature"
```

4. **Push to your fork**
```bash
git push origin feature/your-feature-name
```

5. **Open a Pull Request**
   - Fill out the PR template
   - Link any related issues
   - Request review from maintainers

6. **Address feedback**
   - Make requested changes
   - Push additional commits
   - Re-request review when ready

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for all public functions

### Example:
```python
def train_marine(obs, _x: float = None, _y: float = None) -> actions.FunctionCall:
    """
    Train a Marine from Barracks.
    
    Args:
        obs: PySC2 observation object
        _x: Unused coordinate parameter (for interface consistency)
        _y: Unused coordinate parameter (for interface consistency)
    
    Returns:
        PySC2 action to train a Marine, or no_op if no Barracks available
    """
    barracks = [
        u for u in obs.observation.raw_units
        if u.alliance == features.PlayerRelative.SELF
        and u.unit_type == units.Terran.Barracks
    ]
    if not barracks:
        return actions.RAW_FUNCTIONS.no_op()
    
    return actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks[0].tag)
```

### File Organization

- Keep files focused on single responsibility
- Group related functions together
- Use clear section comments for long files

```python
# ============================================================================
# SECTION NAME
# ============================================================================

def function_in_section():
    pass
```

### Imports

Order imports as:
1. Standard library
2. Third-party packages
3. Local modules

```python
import json
from collections import defaultdict

import numpy as np
import torch
from pysc2.lib import actions, features, units

from game_perception import GamePerception
from terran_helper import distance
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_coordinate_transform.py

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Match test file names to source files (e.g., `test_coordinate_transform.py`)
- Use descriptive test names

```python
def test_coordinate_transform_normalizes_top_left_spawn():
    """Top-left spawn should not need rotation."""
    transform = CoordinateTransform(176, 176, 32, 32)
    assert not transform.needs_rotation
    
    nx, ny = transform.world_to_normalized(32, 32)
    assert abs(nx) < 0.01
    assert abs(ny) < 0.01

def test_coordinate_transform_rotates_bottom_right_spawn():
    """Bottom-right spawn should rotate 180 degrees."""
    transform = CoordinateTransform(176, 176, 144, 144)
    assert transform.needs_rotation
```

### Integration Testing

For changes that affect SC2 interaction:
1. Test with a single worker first
2. Verify the action executes correctly in-game
3. Check for crashes or exceptions

## Documentation

### When to Update Docs

- Adding new features → Update README.md
- Changing architecture → Update ARCHITECTURE.md
- API changes → Update docstrings
- New configuration options → Update README.md

### Docstring Format

Use Google-style docstrings:

```python
def calculate_step_reward(
    self, 
    perception: GamePerception, 
    action_name: str, 
    obs
) -> float:
    """
    Calculate reward for a single step.
    
    Args:
        perception: Current game state perception
        action_name: Name of the action that was executed
        obs: Raw PySC2 observation
    
    Returns:
        Reward value (currently always 0.0 for pure win/loss)
    
    Note:
        This method exists for potential future reward shaping.
        Currently returns 0.0 as we use pure terminal rewards.
    """
    return 0.0
```

## Questions?

- Open an issue for general questions
- Tag maintainers for urgent matters
- Check existing issues and discussions first

Thank you for contributing! 🎮🤖