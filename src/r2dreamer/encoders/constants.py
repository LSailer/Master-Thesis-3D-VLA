"""Shared dimensions for R2Dreamer encoder modules.

The VGGT aggregator emits one camera token, a fixed number of register tokens,
and one token per image patch. ``TokenTransformerEncoder`` needs both counts as
static defaults, and they live here (small ints) rather than being imported from
the adapters so importing an encoder module stays free of the heavy VGGT
extractor dependency. This module is the single source: the token adapters
import ``AGG_REGISTER_TOKENS`` from here to derive their patch slice, so the
encoder that drops the register tokens and the adapter that pools past them
cannot disagree.
"""

AGG_TOKEN_TOKENS = 1374  # cam + registers + patches
AGG_REGISTER_TOKENS = 4
