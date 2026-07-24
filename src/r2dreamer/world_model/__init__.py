"""World-model subpackage: RSSM, head primitives, and world-model loss."""

from .factory import compute_dtype_kwargs, make_rssm
from .rssm import RMSNorm, BlockLinear, Deter, R2RSSM
from .heads import R2MLP, R2TwoHotDist, onehot_mode_st
from .loss import world_model_loss, kl_loss
