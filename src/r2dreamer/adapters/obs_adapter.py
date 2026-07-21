"""ObsAdapter: base class bridging env observations to agent/buffer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable


import jax.numpy as jnp
from src.environments.observation import ObservationFrame

if TYPE_CHECKING:
    from src.r2dreamer.observation_preparation.contracts import PreparedObservation


BufferShape = tuple[int, ...] | Mapping[str, tuple[int, ...]]
BufferDType = str | Mapping[str, str]
BufferNormalize = bool | Mapping[str, bool]


@dataclass
class ObsAdapter:
    """Bridges env observations to agent/buffer, called once per step.

    Default: extracts the image for replay and passes the live image directly to
    the Encoder Module.
    """

    buffer_dtype: BufferDType = "uint8"
    buffer_shape: BufferShape = (3, 64, 64)
    normalize_on_sample: BufferNormalize = True
    agent_obs_shape: BufferShape | None = None
    # Scene-aware episode-boundary hook. The trainer calls it (when set) at
    # every episode reset — prefill start, prefill episode-end, train reset,
    # eval reset — passing the incoming reset frame's ``scene_id`` so a
    # PERSIST-scene extractor can save the outgoing scene and restore the
    # incoming one (see src/prototyp/live_house_context/PROTOCOL.md §2). A
    # no-arg ``extractor.reset`` is no longer sufficient: the callback fires
    # *before* the first frame's ``extract()``, where ``scene_id`` first
    # becomes available, and the prefill loop discards reset frames entirely
    # (so the in-extract ``is_first`` reset path never fires during prefill).
    on_episode_reset: Callable[[str], None] | None = None

    @property
    def encoder_obs_shape(self) -> BufferShape:
        """Shape consumed by the agent encoder after any adapter/batch packing."""
        if self.agent_obs_shape is not None:
            return self.agent_obs_shape
        if isinstance(self.buffer_shape, Mapping):
            raise ValueError(
                "multi-field adapters must set agent_obs_shape for the encoder"
            )
        return self.buffer_shape

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[jnp.ndarray | dict[str, jnp.ndarray], dict]:
        """Returns (buffer_obs, live step observation dict)."""
        return env_obs.image, {"image": env_obs.image, "is_first": env_obs.is_first}

    def prepare_env_step(self, env_obs: ObservationFrame) -> "PreparedObservation":
        """Return the explicit replay/encoder observation pair."""
        prepared_observation_cls = import_module(
            "src.r2dreamer.observation_preparation.contracts"
        ).PreparedObservation
        replay_obs, step_obs = self.transform(env_obs)
        return prepared_observation_cls(
            replay_obs=replay_obs,
            encoder_obs={k: v for k, v in step_obs.items() if k != "is_first"},
            is_first=bool(step_obs["is_first"]),
        )

    def augment_replay_batch(self, batch: Any) -> Any:
        """Optionally add live adapter context to a sampled replay batch."""
        return batch

    def diagnostics(self) -> dict[str, float]:
        """Return adapter health metrics for the end-of-run summary.

        Called once when the run finishes, so implementations may synchronize
        device scalars to host.
        """
        return {}

    @property
    def growth_history(self) -> list[tuple[int, int]]:
        """``(env_step, value)`` time series for the end-of-run summary.

        Companion to :meth:`diagnostics`: adapters that track a growth curve
        (e.g. stored house points at doubling env steps) override this so the
        trainer can persist the series alongside the final stats.
        """
        return []
