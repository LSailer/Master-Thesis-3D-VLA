"""Behavior subpackage: imagination, return EMA, and policy/value losses."""

from .return_ema import ReturnEMA
from .imagination import imagine, lambda_return_positional
from .loss import behavior_loss

__all__ = ["ReturnEMA", "imagine", "lambda_return_positional", "behavior_loss"]
