"""Generic composite encoder: branch-per-obs-key + a fusion strategy.

One Flax module (:class:`CompositeEncoder`) replaces the family of hand-written
combination encoders (``WMHybridEncoder``, ``WP64CNNCPMLPEncoder``,
``HybridHousePointsCameraEncoder``, ``HouseGlobalEmbeddingEncoder``). A
combination is described declaratively by a :class:`CompositeSpec` — an ordered
tuple of :class:`BranchSpec` (which observation key each branch reads and which
mechanism module encodes it) plus a fusion name (``concat`` / ``gate`` /
``concat_mlp``). The spec is static, so the branch loop unrolls at trace time to
exactly the graph the old hand-written class produced.

Golden-run bit-identity is the hard constraint (see the module's parity tests):
Flax folds each parameter's init RNG over the submodule *name* and *depth*, so
the composite must reproduce the legacy parameter tree, not merely the legacy
computation.

- **Single-branch ``concat``** (the CNN encoder) uses :func:`flax.linen.share_scope`
  so the one branch's parameters land at the composite root — byte-identical to
  a bare ``ConvEncoder``, with no wrapping ``image/`` level.
- **Multi-branch** fusions name each branch submodule with its *legacy* Flax
  name (e.g. the hybrid encoder's ``cnn`` / ``vggt_mlp`` and its root ``gate``
  scalar), which differs from the branch's observation key (``image`` /
  ``wp_cp``). ``BranchSpec`` therefore carries both.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import flax.linen as nn
import jax.numpy as jnp


@dataclass(frozen=True)
class BranchSpec:
    """One branch of a :class:`CompositeSpec`.

    Attributes:
        obs_key: Key selecting this branch's slice of the observation dict. For
            a single-branch encoder whose observation is a bare array (e.g. the
            CNN replay image), the array is passed through unchanged when the
            key is absent.
        module_name: Flax submodule name used for this branch's parameter
            subtree. Mirrors the legacy encoder's submodule name (``cnn`` /
            ``vggt_mlp`` / ...) so init RNG folding — and thus the golden run —
            matches. Passed to ``make`` as ``name``. Ignored (``name=None``)
            for a share-scoped single ``concat`` branch, whose parameters live
            at the composite root.
        make: Factory ``make(name) -> nn.Module`` returning a fresh mechanism
            module (``ConvEncoder``, ``R2MLP``, ``PointNet``, ...) constructed
            with the given Flax ``name``.
    """

    obs_key: str
    module_name: str
    make: Callable[[str | None], nn.Module]


@dataclass(frozen=True)
class CompositeSpec:
    """Declarative description of a composite encoder.

    Attributes:
        branches: Ordered branches; the first branch is the ungated backbone
            for ``gate`` fusion. Order fixes the concat layout, so it is part of
            the observable contract.
        fusion: Fusion strategy name — a key of :data:`FUSIONS`.
    """

    branches: tuple[BranchSpec, ...]
    fusion: str = "concat"

    @property
    def branch_keys(self) -> tuple[str, ...]:
        """Return the observation keys this composite reads, in branch order."""
        return tuple(b.obs_key for b in self.branches)


def _select(obs: Any, key: str) -> Any:
    """Return ``obs[key]`` for a dict observation, or the bare array itself.

    A single-branch encoder may receive its observation either as a one-key
    dict (acting/init path) or as a bare array (CNN replay stores the image
    directly); both must reach the mechanism module unchanged.
    """
    if isinstance(obs, Mapping):
        return obs[key]
    return obs


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------
#
# A fusion receives the composite module (so it may create root-level fusion
# parameters, e.g. the gate scalar) and the ordered list of branch outputs, and
# returns the fused embedding. Every fusion preserves leading dims — branches
# operate only on the trailing feature axis.


def _fuse_concat(module: "CompositeEncoder", parts: list[jnp.ndarray]) -> jnp.ndarray:
    """Concatenate branch outputs along the feature axis (no parameters)."""
    del module
    return jnp.concatenate(parts, axis=-1)


def _fuse_gate(module: "CompositeEncoder", parts: list[jnp.ndarray]) -> jnp.ndarray:
    """Ungated backbone plus zero-initialized-gated remaining branches.

    Reproduces ``WMHybridEncoder`` exactly for two branches:
    ``concat([backbone, gate * feature])`` with a single root scalar ``gate``
    starting at 0. Extra branches share the same gate. The gate opens over
    training, so any gain over the backbone is attributable to the extra
    modality rather than architecture drift.
    """
    gate = module.param("gate", nn.initializers.zeros, ())
    fused = [parts[0]] + [gate * part for part in parts[1:]]
    return jnp.concatenate(fused, axis=-1)


def _fuse_concat_mlp(
    module: "CompositeEncoder", parts: list[jnp.ndarray]
) -> jnp.ndarray:
    """Concatenate branches, then project to ``embed_dim`` with one Dense.

    Requires ``CompositeEncoder.embed_dim`` to be set.
    """
    if module.embed_dim is None:
        raise ValueError("concat_mlp fusion requires embed_dim to be set")
    x = jnp.concatenate(parts, axis=-1)
    return nn.Dense(module.embed_dim, name="fusion_proj")(x)


FUSIONS: dict[str, Callable[["CompositeEncoder", list[jnp.ndarray]], jnp.ndarray]] = {
    "concat": _fuse_concat,
    "gate": _fuse_gate,
    "concat_mlp": _fuse_concat_mlp,
}


class CompositeEncoder(nn.Module):
    """Encode a multi-key observation into one embedding via branches + fusion.

    Attributes:
        spec: The static branch/fusion description. Because it is static, the
            branch loop unrolls at trace time — there is no runtime dict lookup
            or Python-level branching inside the compiled graph.
        embed_dim: Target width for the ``concat_mlp`` fusion projection.
            Unused by ``concat`` / ``gate`` (their width is the sum of branch
            widths).
    """

    spec: CompositeSpec
    embed_dim: int | None = None

    @property
    def branches(self) -> tuple[str, ...]:
        """Return the observation keys, for the startup branch-key check."""
        return self.spec.branch_keys

    @nn.compact
    def __call__(self, obs: Any) -> jnp.ndarray:
        """Encode ``obs`` into ``(..., E)``, preserving all leading dims."""
        branches = self.spec.branches
        if not branches:
            raise ValueError("CompositeSpec must declare at least one branch")

        # Single-branch concat: share the composite scope so the branch's
        # parameters land at the root, byte-identical to the bare mechanism
        # module (no wrapping submodule level). This is what keeps the CNN
        # encoder's golden run bit-identical.
        if len(branches) == 1 and self.spec.fusion == "concat":
            bspec = branches[0]
            module = bspec.make(None)
            nn.share_scope(self, module)
            return module(_select(obs, bspec.obs_key))

        parts = [
            bspec.make(bspec.module_name)(_select(obs, bspec.obs_key))
            for bspec in branches
        ]
        try:
            fuse = FUSIONS[self.spec.fusion]
        except KeyError:
            raise ValueError(
                f"unknown fusion {self.spec.fusion!r}; "
                f"expected one of {sorted(FUSIONS)}"
            ) from None
        return fuse(self, parts)
