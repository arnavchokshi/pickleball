"""Scoped event-spotting training and evaluation scaffold.

This package is not a promoted runtime. ``VERIFIED=0`` remains binding.
"""

from .model import EventHead, masked_cross_entropy

__all__ = ["EventHead", "masked_cross_entropy"]
