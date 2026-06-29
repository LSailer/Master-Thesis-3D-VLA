import json
from types import SimpleNamespace

import jax

from src.r2dreamer.launch import evaluate as eval_module
from src.r2dreamer.launch.evaluate import (
    _find_manifest_for_checkpoint,
    _load_arch_overrides_from_manifest,
)
from src.r2dreamer.observation_preparation import CNNObservationPreparation
from src.r2dreamer.observation_preparation.contracts import PreparedObservation
from src.r2dreamer.encoders.cnn import ConvEncoder


def test_find_manifest_next_to_checkpoint(tmp_path):
    ckpt = tmp_path / "step_000000010.pkl"
    manifest = tmp_path / "MANIFEST.json"
    ckpt.touch()
    manifest.write_text('{"config": {}}')

    assert _find_manifest_for_checkpoint(ckpt) == manifest.resolve()


def test_find_manifest_in_run_dir_for_checkpoints_subdir(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "step_000000010.pkl"
    manifest = tmp_path / "MANIFEST.json"
    ckpt.touch()
    manifest.write_text('{"config": {}}')

    assert _find_manifest_for_checkpoint(ckpt) == manifest.resolve()


def test_load_arch_overrides_recovers_encoder_input_contract_from_manifest(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "step_000000010.pkl"
    ckpt.touch()
    manifest = tmp_path / "MANIFEST.json"
    manifest.write_text(json.dumps({
        "config": {
            "encoder_input_contract": CNNObservationPreparation().contract.to_snapshot(),
        }
    }))

    overrides = _load_arch_overrides_from_manifest(str(ckpt))

    assert overrides["obs_shape"] == (3, 64, 64)
    assert overrides["encoder_type"] == "cnn"
    assert overrides["encoder_module_cls"] is ConvEncoder
    assert overrides["encoder_input_contract"]["encoder_module_kwargs"] == {}


def test_run_eval_episode_updates_obs_after_nonterminal_step(monkeypatch, tmp_path):
    class _Position:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class _FakeSim:
        def __init__(self, env):
            self._env = env

        def get_agent_state(self):
            return SimpleNamespace(
                position=_Position([float(self._env.step_count), 0.0, 0.0])
            )

    class _FakeEnv:
        def __init__(self):
            self.step_count = 0
            self._env = SimpleNamespace(sim=_FakeSim(self))

        def step(self, action):
            self.step_count += 1
            return {
                "id": f"step{self.step_count}",
                "done": False,
                "reward": float(action),
                "success": float(self.step_count),
                "spl": float(self.step_count) / 1000.0,
            }

    class _FakeAdapter:
        def transform(self, obs):
            return None, f"agent-{obs['id']}"

        def prepare_env_step(self, obs, _packer):
            _, encoder_obs = self.transform(obs)
            return PreparedObservation(
                replay_obs=None, encoder_obs=encoder_obs, is_first=False
            )

    class _FakeAgent:
        def __init__(self):
            self.seen = []

        def initial_act_state(self):
            return None

        def act_with_state(self, encoder_obs, is_first, state, act_key, training=False):
            self.seen.append(encoder_obs)
            return 1, state

    def _fake_start_episode(env_instance, adapter, _packer):
        obs = {"id": "initial", "done": False, "reward": 0.0, "success": 0.0, "spl": 0.0}
        _, encoder_obs = adapter.transform(obs)
        return (
            obs,
            encoder_obs,
            True,
            [0.0, 0.0, 0.0],
            [],
            "scene",
            "chair",
            [[0.0, 0.0, 0.0]],
            [0.0],
        )

    monkeypatch.setattr(eval_module, "_start_eval_episode", _fake_start_episode)
    monkeypatch.setattr(eval_module, "_get_agent_heading", lambda env_instance: 0.0)

    agent = _FakeAgent()
    result, _ = eval_module._run_eval_episode(
        ep_idx=0,
        args=SimpleNamespace(log_video_episodes=0, render_topdown=False),
        env_instance=_FakeEnv(),
        adapter=_FakeAdapter(),
        agent=agent,
        rng_key=jax.random.PRNGKey(0),
        config=SimpleNamespace(num_actions=4),
        wandb_module=None,
        output_dir=str(tmp_path),
    )

    assert agent.seen[:3] == ["agent-initial", "agent-step1", "agent-step2"]
    assert result["steps"] == 500
    assert result["success"] == 500.0
