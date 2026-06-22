"""Behavior subpackage: imagination, return EMA, and policy/value losses."""

from .return_ema import ReturnEMA
from .imagination import _imagine
from .loss import behavior_loss
