"""
methods/base.py
Abstract base class that all droplet-removal methods must implement.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path


class BaseMethod(ABC):
    """
    Contract for a droplet-removal method.

    Subclasses implement `process()` and optionally `add_args()` to
    register their own CLI parameters.
    """

    #: Short name used on the command line, e.g. "classical"
    name: str = ""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    # ------------------------------------------------------------------
    # Override in subclasses
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        Register method-specific arguments with the shared ArgumentParser.
        Called once at startup before args are parsed.
        Override to add your own flags.
        """

    @abstractmethod
    def process(self, input_path: Path, output_path: Path) -> None:
        """
        Read `input_path`, remove droplets, write result to `output_path`.

        Args:
            input_path: Path to the source MP4.
            output_path: Path where the cleaned MP4 will be written.
        """
