"""Extract VGGT WP/CP + Dreamer RSSM latents for the first-frame invariance probe.

Experiment (see docs/notes / meeting 2026-05-28):
VGGT predicts World Points (WP) and Camera Pose (CP) *relative to the first frame*
of its streaming window. So the same physical viewpoint, reached as frame #1 vs
frame #20, lands in two different reference frames. We test whether this leaks
into the trained Dreamer representation.

For each episode (same HM3D house, different start pose):
  FORWARD  : navigate start -> goal viewpoint with ShortestPathFollower.
  BACKWARD : KEEP the same sim/scene, reset the VGGT KV-cache (new first-frame
             reference) and navigate the SAME physical path back end -> start.

Per frame we record: RGB (downsampled), agent pose (pos + quat), raw VGGT WP
(37,37,3) + CP (9,), the Dreamer encoder embedding (1024), and the RSSM posterior
latent (stoch 32x16, deter 2048).

GPU job — run via scripts/r2dreamer/invariance_extract.sbatch. CPU has no usable
JAX device on the login node.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ACTION_NAMES = {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT"}
CKPT_DEFAULT = "output/r2dreamer-curriculum-l1-vggt/run-4216462/checkpoints/step_002000000.pkl"
CURRICULUM_L1 = "data/curriculum/level1_1house_1goal.json"
OBS_HW = 518
RGB_SAVE_STRIDE = 2  # 518 -> 259 thumbnails for the HTML report


def quat_xyzw(rotation) -> np.ndarray:
    return np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32)


def agent_pose(env):
    """(position(3,), quaternion xyzw(4,)) of the agent root, plus the RGB sensor pose."""
    st = env._env.sim.get_agent_state()
    sensor = st.sensor_states.get("rgb", st)
    return (
        np.asarray(st.position, dtype=np.float32),
        quat_xyzw(st.rotation),
        np.asarray(sensor.position, dtype=np.float32),
        quat_xyzw(sensor.rotation),
    )


def run_segment(env, follower, goal_pos, max_steps, extractor, agent, flatten_fn,
                jnp, jax, rng):
    """Drive one navigation segment, recording per-frame geometry + latents.

    Returns a dict of stacked numpy arrays. Resets the VGGT KV-cache so this
    segment is its own streaming window (its own first-frame reference).
    """
    params = agent.params
    rec = {k: [] for k in (
        "pos", "rot", "sensor_pos", "sensor_rot", "action",
        "wp", "cp", "embed", "stoch", "deter", "logit", "rgb",
    )}

    extractor.reset()  # fresh first-frame reference for this segment
    stoch, deter = agent.rssm_mod.apply(
        params["rssm"], 1, method=agent.rssm_mod.initial_state)
    prev_action = jnp.zeros((1, 4), dtype=jnp.float32)  # no action before frame 0

    for t in range(max_steps):
        img = env._last_obs["rgb"][:, :, :3]                  # (H, W, 3) uint8 HWC
        img_chw = np.transpose(img, (2, 0, 1))                # (3, H, W) for VGGT

        out = extractor.extract(img_chw)
        wp = np.asarray(out["world_points"], dtype=np.float32)   # (37,37,3)
        cp = np.asarray(out["camera_pose"], dtype=np.float32)    # (9,)
        feat = flatten_fn(out)[None]                             # (1, 4116)

        embed = agent.encoder_mod.apply(params["encoder"], feat)  # (1, 1024)
        rng, k = jax.random.split(rng)
        stoch, deter, logit = agent.rssm_mod.apply(
            params["rssm"], stoch, deter, prev_action, embed,
            rngs={"sample": k})

        pos, rot, spos, srot = agent_pose(env)

        # decide next action (this is what *led away* from this frame)
        a = follower.get_next_action(goal_pos)
        a = 0 if a is None else int(a)

        rec["pos"].append(pos);          rec["rot"].append(rot)
        rec["sensor_pos"].append(spos);  rec["sensor_rot"].append(srot)
        rec["action"].append(a)
        rec["wp"].append(wp);            rec["cp"].append(cp)
        rec["embed"].append(np.asarray(embed[0], dtype=np.float32))
        rec["stoch"].append(np.asarray(stoch[0], dtype=np.float32))
        rec["deter"].append(np.asarray(deter[0], dtype=np.float32))
        rec["logit"].append(np.asarray(logit[0], dtype=np.float32))
        rec["rgb"].append(img[::RGB_SAVE_STRIDE, ::RGB_SAVE_STRIDE].copy())

        if a == 0:  # reached goal / follower gives up
            break
        env.step(a)
        prev_action = jax.nn.one_hot(jnp.array([a]), 4).astype(jnp.float32)

    return {k: np.stack(v) for k, v in rec.items()}, rng


def run_replay_segment(env, fwd, extractor, agent, flatten_fn, jnp, jax, rng):
    """Teleport-replay: revisit the forward poses in REVERSE order at the EXACT
    same position+rotation (identical heading), with a fresh VGGT window.

    Isolates the first-frame reference effect from the heading/content change
    that physical backward navigation introduces. Matched 1:1 to forward by pose.
    The action fed to the RSSM is the forward action at the visited frame (a
    proxy — teleport has no real transition); the latent comparison therefore
    still carries a reversed-history confound, but WP/CP/embed are clean.
    """
    import quaternion
    from habitat_sim.agent import AgentState

    params = agent.params
    sim = env._env.sim
    agent_h = sim.get_agent(0)
    rec = {k: [] for k in (
        "pos", "rot", "sensor_pos", "sensor_rot", "action",
        "wp", "cp", "embed", "stoch", "deter", "logit", "rgb",
    )}

    extractor.reset()
    stoch, deter = agent.rssm_mod.apply(
        params["rssm"], 1, method=agent.rssm_mod.initial_state)
    prev_action = jnp.zeros((1, 4), dtype=jnp.float32)

    for i in range(len(fwd["pos"]))[::-1]:
        p = fwd["pos"][i]
        r = fwd["rot"][i]  # xyzw
        agent_h.set_state(AgentState(
            position=np.asarray(p, dtype=np.float32),
            rotation=quaternion.quaternion(float(r[3]), float(r[0]), float(r[1]), float(r[2])),
        ))
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"] if "rgb" in obs else next(iter(obs.values()))
        img = np.asarray(rgb)[:, :, :3]            # (H, W, 3) HWC uint8
        img_chw = np.transpose(img, (2, 0, 1))

        out = extractor.extract(img_chw)
        wp = np.asarray(out["world_points"], dtype=np.float32)
        cp = np.asarray(out["camera_pose"], dtype=np.float32)
        feat = flatten_fn(out)[None]

        embed = agent.encoder_mod.apply(params["encoder"], feat)
        rng, k = jax.random.split(rng)
        stoch, deter, logit = agent.rssm_mod.apply(
            params["rssm"], stoch, deter, prev_action, embed, rngs={"sample": k})

        pos, rot, spos, srot = agent_pose(env)
        a = int(fwd["action"][i])
        rec["pos"].append(pos);          rec["rot"].append(rot)
        rec["sensor_pos"].append(spos);  rec["sensor_rot"].append(srot)
        rec["action"].append(a)
        rec["wp"].append(wp);            rec["cp"].append(cp)
        rec["embed"].append(np.asarray(embed[0], dtype=np.float32))
        rec["stoch"].append(np.asarray(stoch[0], dtype=np.float32))
        rec["deter"].append(np.asarray(deter[0], dtype=np.float32))
        rec["logit"].append(np.asarray(logit[0], dtype=np.float32))
        rec["rgb"].append(img[::RGB_SAVE_STRIDE, ::RGB_SAVE_STRIDE].copy())
        prev_action = jax.nn.one_hot(jnp.array([a]), 4).astype(jnp.float32)

    return {k: np.stack(v) for k, v in rec.items()}, rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=CKPT_DEFAULT)
    ap.add_argument("--curriculum", default=CURRICULUM_L1)
    ap.add_argument("--num-episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--min-geodesic", type=float, default=4.0,
                    help="skip episodes whose start->goal geodesic is shorter (m)")
    ap.add_argument("--max-reset-tries", type=int, default=60)
    ap.add_argument("--goal-radius", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mode", choices=["navigate", "teleport"], default="navigate",
                    help="backward run: physical navigation, or pose-replay (same heading)")
    ap.add_argument("--out-dir", default="output/analysis/invariance")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[invariance] device={args.device} ckpt={args.checkpoint}", flush=True)

    import jax
    import jax.numpy as jnp
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    from src.environments.habitat import build_habitat_env
    from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor
    from src.r2dreamer.adapters.vggt_adapter import flatten_world_points_camera_pose
    from src.r2dreamer.composition import learner_from_checkpoint, make_learner
    from src.r2dreamer.encoders.mlp import MLPEncoder

    print(f"[invariance] jax devices: {jax.devices()}", flush=True)

    t0 = time.time()
    extractor = JAXVGGTFeatureExtractor(
        device=args.device, total_budget=200_000,
        budgets_static=tuple([8333] * 24), compute_heads=True,
    )
    print(f"[invariance] VGGT extractor ready ({time.time()-t0:.1f}s)", flush=True)

    agent = learner_from_checkpoint(
        args.checkpoint, obs_shape=(4116,), num_actions=4, seed=args.seed,
        encoder_type="vggt", encoder_module_cls=MLPEncoder,
    )
    print(f"[invariance] agent loaded, checkpoint_step={agent.checkpoint_step}", flush=True)

    env = build_habitat_env(
        obs_shape=(OBS_HW, OBS_HW, 3), curriculum_path=args.curriculum,
        mode="train", seed=args.seed,
    )
    follower = ShortestPathFollower(env._env.sim, args.goal_radius, return_one_hot=False)

    rng = jax.random.PRNGKey(args.seed)
    manifest = {
        "checkpoint": args.checkpoint, "curriculum": args.curriculum,
        "checkpoint_step": agent.checkpoint_step, "max_steps": args.max_steps,
        "min_geodesic": args.min_geodesic, "seed": args.seed, "mode": args.mode,
        "episodes": [],
    }

    collected = 0
    tries = 0
    while collected < args.num_episodes and tries < args.max_reset_tries:
        tries += 1
        env.reset()
        geo = float(env._start_geodesic)
        if not np.isfinite(geo) or geo < args.min_geodesic:
            continue

        goal_pos, _ = env.find_nearest_viewpoint()
        if goal_pos is None:
            continue
        goal_pos = np.asarray(goal_pos, dtype=np.float32)
        start_pos = np.asarray(env._env.sim.get_agent_state().position, dtype=np.float32)
        ep = env._env.current_episode
        ep_id, scene = ep.episode_id, ep.scene_id.split("/")[-1]

        print(f"[invariance] episode {collected} id={ep_id} geo={geo:.2f}m -> forward",
              flush=True)
        fwd, rng = run_segment(env, follower, goal_pos, args.max_steps,
                               extractor, agent, flatten_world_points_camera_pose,
                               jnp, jax, rng)

        # BACKWARD: same scene, fresh VGGT window. Either physically navigate back
        # to start (navigate) or replay the forward poses in reverse (teleport).
        print(f"[invariance] episode {collected} -> backward[{args.mode}] "
              f"({len(fwd['action'])} fwd steps)", flush=True)
        if args.mode == "teleport":
            bwd, rng = run_replay_segment(env, fwd, extractor, agent,
                                          flatten_world_points_camera_pose, jnp, jax, rng)
        else:
            bwd, rng = run_segment(env, follower, start_pos, args.max_steps,
                                   extractor, agent, flatten_world_points_camera_pose,
                                   jnp, jax, rng)

        ep_path = out_dir / f"episode_{collected:02d}.npz"
        np.savez_compressed(
            ep_path,
            **{f"fwd_{k}": v for k, v in fwd.items()},
            **{f"bwd_{k}": v for k, v in bwd.items()},
            start_pos=start_pos, goal_pos=goal_pos, geodesic=np.float32(geo),
        )
        manifest["episodes"].append({
            "index": collected, "episode_id": ep_id, "scene": scene,
            "geodesic": geo, "fwd_steps": int(len(fwd["action"])),
            "bwd_steps": int(len(bwd["action"])), "file": ep_path.name,
        })
        print(f"[invariance] saved {ep_path.name}", flush=True)
        collected += 1

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[invariance] DONE: {collected} episodes in {out_dir} "
          f"({time.time()-t0:.1f}s total)", flush=True)
    if collected < args.num_episodes:
        print(f"[invariance] WARNING: only {collected}/{args.num_episodes} episodes "
              f"met min_geodesic={args.min_geodesic}m after {tries} resets", flush=True)
    env.close()


if __name__ == "__main__":
    main()
