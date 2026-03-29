"""Run PyTorch r2dreamer on Crafter with a standalone training loop.

Bypasses r2dreamer's trainer/buffer (which need torchrl) by using the Dreamer
agent class directly with our own simple replay buffer and Crafter env.

Supports both rep_loss=dreamer (DreamerV3 with decoder) and rep_loss=r2dreamer.
Outputs metrics to CSV for comparison with the JAX implementation.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from torch.amp import autocast

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
R2DREAMER_DIR = os.path.join(REPO_ROOT, "external", "r2dreamer")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, R2DREAMER_DIR)


# ---------------------------------------------------------------------------
# Simple numpy replay buffer (no torchrl dependency)
# ---------------------------------------------------------------------------

class SimpleReplayBuffer:
    """Ring buffer storing transitions as numpy arrays, sampling contiguous sequences."""

    def __init__(self, capacity, obs_shape_hwc, num_actions, stoch_shape, deter_size):
        H, W, C = obs_shape_hwc
        self.capacity = capacity
        self.num_actions = num_actions
        self.stoch_shape = stoch_shape  # (S, K)
        self.deter_size = deter_size

        self.image = np.zeros((capacity, H, W, C), dtype=np.uint8)
        self.action = np.zeros((capacity, num_actions), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.is_first = np.zeros((capacity, 1), dtype=bool)
        self.is_last = np.zeros((capacity, 1), dtype=bool)
        self.is_terminal = np.zeros((capacity, 1), dtype=bool)
        # Stored latent states for RSSM warm-start
        self.stoch = np.zeros((capacity, *stoch_shape), dtype=np.float32)
        self.deter = np.zeros((capacity, deter_size), dtype=np.float32)
        self.episode_id = np.zeros(capacity, dtype=np.int64)

        self.idx = 0
        self.size = 0
        self._current_ep = 0

    def add(self, image, action_onehot, reward, is_first, is_last, is_terminal):
        i = self.idx
        self.image[i] = image
        self.action[i] = action_onehot
        self.reward[i] = reward
        self.is_first[i] = is_first
        self.is_last[i] = is_last
        self.is_terminal[i] = is_terminal
        self.episode_id[i] = self._current_ep
        if is_last:
            self._current_ep += 1
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, seq_len, device):
        """Sample contiguous sequences, return (data_dict, initial_tuple)."""
        from tensordict import TensorDict

        # Sample starting indices that don't cross episode boundaries
        sequences = []
        attempts = 0
        while len(sequences) < batch_size and attempts < batch_size * 20:
            start = np.random.randint(0, self.size - seq_len)
            # Check no episode boundary in the middle (is_first=True after start)
            is_first_slice = self.is_first[start + 1 : start + seq_len, 0]
            if not np.any(is_first_slice):
                sequences.append(start)
            attempts += 1

        # Fallback: if we can't find clean sequences, allow boundaries
        while len(sequences) < batch_size:
            start = np.random.randint(0, self.size - seq_len)
            sequences.append(start)

        starts = np.array(sequences)
        indices = starts[:, None] + np.arange(seq_len)[None, :]  # (B, T)

        dev = torch.device(device)

        # Build data TensorDict (B, T, ...)
        data = TensorDict({
            "image": torch.from_numpy(self.image[indices]).to(dev),
            "action": torch.from_numpy(self.action[indices]).to(dev),
            "reward": torch.from_numpy(self.reward[indices]).to(dev, dtype=torch.float32),
            "is_first": torch.from_numpy(self.is_first[indices]).to(dev),
            "is_last": torch.from_numpy(self.is_last[indices]).to(dev),
            "is_terminal": torch.from_numpy(self.is_terminal[indices]).to(dev),
        }, batch_size=(batch_size, seq_len))

        # Initial latent state: use stored stoch/deter from first timestep
        initial = (
            torch.from_numpy(self.stoch[starts]).to(dev),
            torch.from_numpy(self.deter[starts]).to(dev),
        )

        return data, starts, initial

    def update_latents(self, starts, seq_len, stoch, deter):
        """Write computed latent states back into buffer."""
        stoch_np = stoch.detach().cpu().numpy()
        deter_np = deter.detach().cpu().numpy()
        B, T = stoch_np.shape[0], stoch_np.shape[1]
        for b in range(B):
            for t in range(T):
                idx = (starts[b] + t) % self.capacity
                self.stoch[idx] = stoch_np[b, t]
                self.deter[idx] = deter_np[b, t]


# ---------------------------------------------------------------------------
# Crafter environment (HWC output for PyTorch r2dreamer)
# ---------------------------------------------------------------------------

class CrafterEnvHWC:
    """Crafter env returning HWC uint8 images (r2dreamer convention)."""

    def __init__(self, size=(64, 64), seed=None):
        import crafter
        self._env = crafter.Env(size=size, reward=True, seed=seed)
        self.num_actions = self._env.action_space.n  # 17

    def reset(self):
        obs = self._env.reset()  # (H, W, C) uint8
        return {
            "image": obs,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "reward": 0.0,
        }

    def step(self, action_int):
        obs, reward, done, info = self._env.step(action_int)
        return {
            "image": obs,
            "reward": float(reward),
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
        }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def create_agent(rep_loss, device_str, num_actions):
    """Create r2dreamer Dreamer agent using Hydra config."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    import gym

    GlobalHydra.instance().clear()
    config_dir = os.path.join(R2DREAMER_DIR, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        full_cfg = compose(
            config_name="configs",
            overrides=[
                "env=crafter",
                "model=size12M",
                f"model.rep_loss={rep_loss}",
                f"device={device_str}",
                "model.compile=False",
            ],
        )
    cfg = full_cfg.model

    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
    })
    act_space = gym.spaces.Discrete(num_actions)

    from dreamer import Dreamer
    agent = Dreamer(cfg, obs_space, act_space).to(device_str)
    return agent, cfg


def train(agent, cfg, env, buffer, steps, device_str, log_every, writer):
    """Run the training loop."""
    batch_size = 16
    seq_len = 64
    train_ratio = 512
    batch_frames = batch_size * seq_len

    obs = env.reset()
    state = agent.get_initial_state(1)

    episode_reward = 0.0
    episode_count = 0
    train_credit = 0.0
    t0 = time.time()
    metrics = {}

    for step in range(steps):
        # --- Act ---
        obs_torch = {
            "image": torch.from_numpy(obs["image"][None]).to(device_str),
            "is_first": torch.tensor([obs["is_first"]], device=device_str),
        }
        with torch.no_grad():
            action, state = agent.act(obs_torch, state)

        action_np = action[0].cpu().numpy()  # (num_actions,) one-hot
        action_int = int(np.argmax(action_np))

        # Store transition
        buffer.add(
            image=obs["image"],
            action_onehot=action_np,
            reward=obs["reward"],
            is_first=obs["is_first"],
            is_last=obs["is_last"],
            is_terminal=obs["is_terminal"],
        )

        # Step env
        next_obs = env.step(action_int)
        episode_reward += next_obs["reward"]

        if next_obs["is_last"]:
            # Store terminal transition
            buffer.add(
                image=next_obs["image"],
                action_onehot=np.zeros(env.num_actions, dtype=np.float32),
                reward=next_obs["reward"],
                is_first=False,
                is_last=True,
                is_terminal=True,
            )
            episode_count += 1
            writer.writerow([step, "episode/score", episode_reward])
            episode_reward = 0.0
            obs = env.reset()
            state = agent.get_initial_state(1)
        else:
            obs = next_obs

        # --- Train ---
        if buffer.size >= batch_frames + seq_len:
            train_credit += train_ratio / batch_frames
            while train_credit >= 1.0:
                data, starts, initial = buffer.sample(batch_size, seq_len, device_str)
                # Run full update step
                p_data = agent.preprocess(data)
                agent._update_slow_target()
                with autocast(device_type=torch.device(device_str).type, dtype=torch.float16):
                    (stoch, deter), mets = agent._cal_grad(p_data, initial)
                agent._scaler.unscale_(agent._optimizer)
                agent._agc(agent._named_params.values())
                agent._scaler.step(agent._optimizer)
                agent._scaler.update()
                agent._scheduler.step()
                agent._optimizer.zero_grad(set_to_none=True)

                # Write latents back
                buffer.update_latents(starts, seq_len, stoch, deter)

                metrics = {k: float(v) if torch.is_tensor(v) else float(v)
                           for k, v in mets.items()}
                train_credit -= 1.0

            if step % log_every == 0 and metrics:
                for k, v in metrics.items():
                    writer.writerow([step, k, v])

                elapsed = time.time() - t0
                fps = (step + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"[step {step:>6d}/{steps}] "
                    f"loss={metrics.get('opt/loss', 0):.3f} "
                    f"dyn={metrics.get('loss/dyn', 0):.3f} "
                    f"rew={metrics.get('loss/rew', 0):.3f} "
                    f"fps={fps:.0f}"
                )

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Episodes: {episode_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--prefill", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rep_loss", choices=["dreamer", "r2dreamer"], default="r2dreamer")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = CrafterEnvHWC(size=(64, 64), seed=args.seed)
    agent, cfg = create_agent(args.rep_loss, args.device, env.num_actions)

    # RSSM dimensions from config
    stoch_classes = int(cfg.rssm.stoch)
    stoch_discrete = int(cfg.rssm.discrete)
    deter_size = int(cfg.rssm.deter)

    buffer = SimpleReplayBuffer(
        capacity=500_000,
        obs_shape_hwc=(64, 64, 3),
        num_actions=env.num_actions,
        stoch_shape=(stoch_classes, stoch_discrete),
        deter_size=deter_size,
    )

    # Prefill with random actions
    print(f"Prefilling {args.prefill} steps...")
    obs = env.reset()
    for _ in range(args.prefill):
        action_int = np.random.randint(0, env.num_actions)
        action_onehot = np.zeros(env.num_actions, dtype=np.float32)
        action_onehot[action_int] = 1.0
        buffer.add(obs["image"], action_onehot, obs["reward"],
                    obs["is_first"], obs["is_last"], obs["is_terminal"])
        next_obs = env.step(action_int)
        if next_obs["is_last"]:
            buffer.add(next_obs["image"], np.zeros(env.num_actions, dtype=np.float32),
                        next_obs["reward"], False, True, True)
            obs = env.reset()
        else:
            obs = next_obs

    # Training
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])

        print(f"Training PyTorch r2dreamer ({args.rep_loss}) for {args.steps} steps...")
        train(agent, cfg, env, buffer, args.steps, args.device, args.log_every, writer)

    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
