"""Behavior subpackage: imagination, return EMA, and policy/value losses."""

from .return_ema import ReturnEMA
from .imagination import _imagine, _lambda_return
from .loss import behavior_loss

__all__ = ["ReturnEMA", "_imagine", "_lambda_return", "behavior_loss"]
