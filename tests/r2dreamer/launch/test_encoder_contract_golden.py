"""Golden fixtures pinning the durable encoder contract.

These fixtures are the acceptance test for the ``fold-vggt-specs-onto-encoders``
change. Checkpoints persist ``module_class_path(encoder_module_cls)`` — a
literal import path — inside ``EncoderInputContract.to_snapshot()``, and
``from_snapshot()`` recovers it with ``_import_class``. Any Flax ``nn.Module``
that changes file or qualname silently invalidates every checkpoint that
references it, and the breakage only surfaces when ``evaluate`` tries to load.

So the contract is captured here, on ``main``, before the refactor edits any
production file. Afterwards the payload must regenerate byte-identically. The
fixtures are written and compared through one canonical serializer
(:func:`_dumps`), so per-encoder equality plus key-set equality is byte
equality; :func:`test_golden_fixture_is_canonical` pins that directly and also
catches a fixture edited by hand rather than regenerated.

Nine of the eighteen registry entries expose no ``contract_snapshot`` (their
adapters carry no ``contract``), so ``spec()`` falls back to its own fields.
Both forms are recorded: ``spec`` for every encoder, ``contract_snapshot`` where
one exists and ``null`` otherwise.

Regenerate after an intended change with:

    REGEN_ENCODER_GOLDEN=1 uv run pytest \
        tests/r2dreamer/launch/test_encoder_contract_golden.py

and review the resulting diff — a diff here means the durable contract moved.
"""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from src.configs.agent_config import R2DreamerConfig
from src.r2dreamer.encoders import Encoder, EncoderSpec
from src.r2dreamer.launch.parser import _build_parser_train
from src.r2dreamer.launch.registries import encoder_registry
from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    _import_class,
    module_class_path,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_CONTRACT_FIXTURE = _FIXTURES / "encoder_contract_golden.json"
_KWARGS_FIXTURE = _FIXTURES / "encoder_module_kwargs_golden.json"

_REGEN = os.environ.get("REGEN_ENCODER_GOLDEN") == "1"

_ENCODER_KEYS = sorted(encoder_registry)


class _StubExtractor:
    """Stand-in for ``JAXVGGTFeatureExtractor`` so the fixture needs no GPU.

    Mirrors the real extractor's shape-bearing attributes, which is all the
    adapters read while resolving a contract. Values match the defaults the
    equivalent stub in ``test_encoders.py`` asserts against.
    """

    aggregator_feature_shape = (1374, 1024)
    image_size = 518

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.wp_pool_size = int(kwargs.get("wp_pool_size", 37))

    def reset(self):
        pass

    def reset_for_scene(self, scene_id="scene"):
        del scene_id


@pytest.fixture(autouse=True)
def _stub_vggt(monkeypatch):
    """Install :class:`_StubExtractor` for every test in this module."""
    monkeypatch.setattr(
        "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor", _StubExtractor
    )


def _dumps(payload: Any) -> str:
    """Serialize a payload canonically, so equal payloads are equal bytes.

    Args:
      payload: Any JSON-serializable value.

    Returns:
      The canonical JSON text, sorted by key and newline-terminated.
    """
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _shape_payload(
    shape: tuple[int, ...] | Mapping[str, tuple[int, ...]],
) -> list[int] | dict[str, list[int]]:
    """Render a flat or structured observation shape as JSON."""
    if isinstance(shape, Mapping):
        return {key: list(dims) for key, dims in shape.items()}
    return list(shape)


def _spec_payload(spec: EncoderSpec) -> dict[str, Any]:
    """Render the durable fields of an ``EncoderSpec`` as JSON.

    ``module_cls`` becomes its import path — the same form
    ``to_snapshot()`` persists — so a relocated ``nn.Module`` shows up as a
    fixture diff for the fallback entries too, not only the snapshot ones.

    Args:
      spec: A resolved encoder spec.

    Returns:
      A JSON-serializable mapping of the spec's durable fields.
    """
    return {
        "encoder_type": spec.encoder_type,
        "env_render_resolution": spec.env_render_resolution,
        "encoder_module": module_class_path(spec.module_cls),
        "obs_shape": _shape_payload(spec.obs_shape),
        "agent_overrides": dict(spec.agent_overrides),
        "design_notes": spec.design_notes,
    }


def _resolve_spec(encoder_cls: type[Encoder]) -> EncoderSpec:
    """Build a selection from default train args and resolve its spec."""
    args = _build_parser_train().parse_args([])
    return encoder_cls.from_train_args(args).spec()


def _contract_payload(encoder_cls: type[Encoder]) -> dict[str, Any]:
    """Capture one encoder's durable contract as JSON.

    Args:
      encoder_cls: A launcher-side ``Encoder`` selection class.

    Returns:
      ``spec`` (always) and ``contract_snapshot`` (``None`` where the adapter
      exposes no contract).
    """
    spec = _resolve_spec(encoder_cls)
    return {
        "spec": _spec_payload(spec),
        "contract_snapshot": spec.contract_snapshot,
    }


def _effective_config(spec: EncoderSpec) -> R2DreamerConfig:
    """Build the config the launcher would hand the encoder kwargs formula.

    Mirrors ``_make_agent_config`` (``launch/train.py``): the selection's
    ``agent_overrides`` are applied on top of the config defaults, which is
    what makes e.g. ``vggt_house_context`` resolve at ``vggt_token_dim=2048``
    rather than the bare default. Run-shaped knobs (steps, seed, logdir) are
    omitted — they do not reach the encoder kwargs.

    Args:
      spec: A resolved encoder spec.

    Returns:
      The effective agent config for kwargs resolution.
    """
    return R2DreamerConfig(
        encoder_type=spec.encoder_type,
        encoder_module_cls=spec.module_cls,
        obs_shape=spec.obs_shape,
        **dict(spec.agent_overrides),
    )


def _kwargs_payload(encoder_cls: type[Encoder]) -> dict[str, Any]:
    """Resolve one encoder's module kwargs from its effective config."""
    spec = _resolve_spec(encoder_cls)
    return encoder_cls.module_kwargs_from_config(_effective_config(spec))


def _build_contract_fixture() -> dict[str, Any]:
    return {key: _contract_payload(encoder_registry[key]) for key in _ENCODER_KEYS}


def _build_kwargs_fixture() -> dict[str, Any]:
    return {key: _kwargs_payload(encoder_registry[key]) for key in _ENCODER_KEYS}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        pytest.fail(
            f"missing golden fixture {path.name}; regenerate with "
            f"REGEN_ENCODER_GOLDEN=1 uv run pytest {Path(__file__).name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def _regenerate(request):
    """Rewrite both fixtures when ``REGEN_ENCODER_GOLDEN=1``.

    Module-scoped and ordered before the tests, so a regenerating run also
    verifies what it just wrote. The autouse ``_stub_vggt`` fixture is
    function-scoped and so cannot be relied on here; the stub is installed
    directly for the duration of the rewrite.
    """
    if not _REGEN:
        return
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor", _StubExtractor
    )
    request.addfinalizer(monkeypatch.undo)
    _CONTRACT_FIXTURE.write_text(_dumps(_build_contract_fixture()), encoding="utf-8")
    _KWARGS_FIXTURE.write_text(_dumps(_build_kwargs_fixture()), encoding="utf-8")


class TestContractGolden:
    """Task 1.1 / 1.2 — the durable contract, pinned and round-tripped."""

    def test_golden_fixture_covers_exactly_the_registry(self):
        assert set(_load(_CONTRACT_FIXTURE)) == set(_ENCODER_KEYS)

    @pytest.mark.parametrize("encoder_type", _ENCODER_KEYS)
    def test_contract_payload_matches_golden(self, encoder_type):
        # Parametrized for a readable per-encoder diff; the byte-level
        # guarantee is test_golden_fixture_is_canonical below.
        expected = _load(_CONTRACT_FIXTURE)[encoder_type]
        actual = json.loads(_dumps(_contract_payload(encoder_registry[encoder_type])))

        assert actual == expected, (
            f"{encoder_type}: durable contract changed. If intended, regenerate "
            f"with REGEN_ENCODER_GOLDEN=1 and review the diff — a changed "
            f"encoder_module path invalidates existing checkpoints."
        )

    def test_golden_fixture_is_canonical(self):
        # The acceptance assertion of the change: byte equality, and a guard
        # against a fixture hand-edited to make the tests above pass.
        assert _dumps(_build_contract_fixture()) == _CONTRACT_FIXTURE.read_text(
            encoding="utf-8"
        )

    @pytest.mark.parametrize("encoder_type", _ENCODER_KEYS)
    def test_recorded_encoder_module_still_resolves(self, encoder_type):
        # This is the check that catches a relocated nn.Module: the path in
        # the fixture is what a checkpoint holds, and _import_class is how
        # from_snapshot recovers it.
        record = _load(_CONTRACT_FIXTURE)[encoder_type]
        path = record["spec"]["encoder_module"]

        resolved = _import_class(path)

        assert module_class_path(resolved) == path

    @pytest.mark.parametrize("encoder_type", _ENCODER_KEYS)
    def test_snapshot_round_trips(self, encoder_type):
        snapshot = _load(_CONTRACT_FIXTURE)[encoder_type]["contract_snapshot"]
        if snapshot is None:
            pytest.skip(f"{encoder_type} exposes no contract snapshot")

        contract = EncoderInputContract.from_snapshot(snapshot)

        assert module_class_path(contract.encoder_module_cls) == snapshot[
            "encoder_module"
        ]
        assert contract.encoder_type == snapshot["encoder_type"]


class TestModuleKwargsGolden:
    """Task 1.4 — the two ``encoder_type ==`` if-chains, pinned by their output."""

    def test_golden_fixture_covers_exactly_the_registry(self):
        assert set(_load(_KWARGS_FIXTURE)) == set(_ENCODER_KEYS)

    @pytest.mark.parametrize("encoder_type", _ENCODER_KEYS)
    def test_resolved_kwargs_match_golden(self, encoder_type):
        expected = _load(_KWARGS_FIXTURE)[encoder_type]
        actual = json.loads(_dumps(_kwargs_payload(encoder_registry[encoder_type])))

        assert actual == expected, (
            f"{encoder_type}: resolved module kwargs changed. Collapsing the "
            f"if-chains must be behaviour-preserving; regenerate only for an "
            f"intended change."
        )

    def test_golden_fixture_is_canonical(self):
        assert _dumps(_build_kwargs_fixture()) == _KWARGS_FIXTURE.read_text(
            encoding="utf-8"
        )
