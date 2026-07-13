"""
methods/__init__.py
Registry of available droplet-removal methods.

To add a new method:
  1. Create methods/my_method.py with a class that subclasses BaseMethod
     and sets `name = "my_method"`.
  2. Import it here and add it to REGISTRY.
"""

from __future__ import annotations

import argparse
from typing import Type

from .base import BaseMethod
from .classical import ClassicalMethod

# ------------------------------------------------------------
# Registry: name -> class
# All entries here are exposed as valid --method choices.
# ------------------------------------------------------------
REGISTRY: dict[str, Type[BaseMethod]] = {
    ClassicalMethod.name: ClassicalMethod,
}


def get_method(name: str) -> Type[BaseMethod]:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[name]


def register_all_args(parser: argparse.ArgumentParser) -> None:
    """Let every registered method add its own CLI arguments."""
    for cls in REGISTRY.values():
        cls.add_args(parser)
