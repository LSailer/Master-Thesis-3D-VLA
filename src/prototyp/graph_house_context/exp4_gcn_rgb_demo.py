"""Experiment 4: sparse GCN RGB reconstruction on the house graph.

Proof that the attributed house graph (nodes = xyz, attribute = rgb) is
learnable: corrupt the RGB signal (mask a fraction of nodes, or add noise),
then train a small sparse GCN (UvA JAX tutorial 7 adapted to segment_sum
message passing) to reconstruct it from the neighborhood structure. If the
graph captures the scene's spatial relations, corrupted nodes recover their
color from neighbors — the learning-based counterpart of exp3's fixed GFT
low-pass prior.

Run (CPU login node; subsample keeps it fast):
    JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp4_gcn_rgb_demo \
        --steps 100 --max-points 50000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.prototype_helpers.graph_gcn import RgbGcnAutoencoder
from src.prototype_helpers.graph_metrics import rgb_psnr
from src.prototype_helpers.knn_graph import build_knn_graph
from src.prototype_helpers.ply_io import load_ply_xyzrgb, save_ply_xyzrgb

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLY = (
    REPO_ROOT
    / "output/bench/house_context_50steps/bench_50steps_full_1cm/step_00000_context.ply"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/prototype/graph_house_context/exp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--task", choices=("mask", "noise"), default="mask")
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="0 = full cloud; otherwise even-stride subsample for speed",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb = load_ply_xyzrgb(args.ply)
    if args.max_points and xyz.shape[0] > args.max_points:
        xyzrgb = jnp.concatenate(
            [xyz, jnp.asarray(rgb, dtype=jnp.float32) / 255.0], axis=-1
        )
        rows = HouseContextPoseBuffer.resample_xyzrgb(xyzrgb, args.max_points)
        xyz = rows[:, :3]
        rgb = jnp.asarray(
            jnp.rint(jnp.clip(rows[:, 3:], 0.0, 1.0) * 255.0), dtype=jnp.uint8
        )
    num_nodes = int(xyz.shape[0])
    print(f"training on {num_nodes} points from {args.ply}")

    graph = build_knn_graph(xyz, k=args.k, cuda=args.cuda)
    rgb01 = jnp.asarray(rgb, dtype=jnp.float32) / 255.0
    center = jnp.mean(xyz, axis=0)
    scale = jnp.maximum(jnp.std(xyz, axis=0), 1e-6)
    xyz_normalized = (xyz - center) / scale

    key = jax.random.PRNGKey(args.seed)
    key, corrupt_key, init_key = jax.random.split(key, 3)
    if args.task == "mask":
        corrupted_mask = jax.random.bernoulli(
            corrupt_key, args.mask_frac, (num_nodes,)
        )
        corrupted_rgb = jnp.where(corrupted_mask[:, None], 0.0, rgb01)
    else:
        corrupted_mask = jnp.ones((num_nodes,), dtype=bool)
        noise = args.noise_std * jax.random.normal(corrupt_key, rgb01.shape)
        corrupted_rgb = jnp.clip(rgb01 + noise, 0.0, 1.0)

    node_feats = jnp.concatenate(
        [xyz_normalized, corrupted_rgb, corrupted_mask[:, None].astype(jnp.float32)],
        axis=-1,
    )

    model = RgbGcnAutoencoder(hidden=args.hidden, num_layers=args.layers)
    params = model.init(
        init_key, node_feats, graph.senders, graph.receivers, graph.weights, num_nodes
    )
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        def loss_fn(params):
            prediction = model.apply(
                params,
                node_feats,
                graph.senders,
                graph.receivers,
                graph.weights,
                num_nodes,
            )
            per_node = jnp.sum((prediction - rgb01) ** 2, axis=-1)
            weight = corrupted_mask.astype(jnp.float32)
            return jnp.sum(per_node * weight) / jnp.maximum(jnp.sum(weight), 1.0)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses: list[float] = []
    start = time.perf_counter()
    for step in range(args.steps):
        params, opt_state, loss = train_step(params, opt_state)
        losses.append(float(loss))
        if step == 0 or (step + 1) % 50 == 0:
            print(f"step {step + 1:>5}: loss {losses[-1]:.6f}")
    train_seconds = time.perf_counter() - start

    prediction = model.apply(
        params, node_feats, graph.senders, graph.receivers, graph.weights, num_nodes
    )
    corrupted_nodes = np.asarray(corrupted_mask)
    if not corrupted_nodes.any():
        raise ValueError(
            "no corrupted nodes to evaluate (mask_frac too small?); "
            "PSNR over an empty set is undefined"
        )
    true_255 = np.asarray(rgb01) * 255.0
    predicted_255 = np.asarray(prediction) * 255.0
    corrupted_255 = np.asarray(corrupted_rgb) * 255.0
    psnr_prediction = rgb_psnr(
        true_255[corrupted_nodes], predicted_255[corrupted_nodes]
    )
    psnr_corrupted_input = rgb_psnr(
        true_255[corrupted_nodes], corrupted_255[corrupted_nodes]
    )
    print(
        f"PSNR on corrupted nodes: prediction {psnr_prediction:.2f} dB vs "
        f"corrupted input {psnr_corrupted_input:.2f} dB"
    )

    metrics = {
        "ply": str(args.ply),
        "num_points": num_nodes,
        "task": args.task,
        "mask_frac": args.mask_frac,
        "noise_std": args.noise_std,
        "k": graph.k,
        "hidden": args.hidden,
        "layers": args.layers,
        "steps": args.steps,
        "lr": args.lr,
        "train_seconds": train_seconds,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "psnr_prediction_db": psnr_prediction,
        "psnr_corrupted_input_db": psnr_corrupted_input,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(losses)
    axis.set_xlabel("step")
    axis.set_ylabel("masked MSE loss")
    axis.set_yscale("log")
    axis.set_title(f"GCN RGB reconstruction ({args.task})")
    figure.tight_layout()
    figure.savefig(args.out_dir / "loss_curve.png", dpi=120)
    plt.close(figure)

    save_ply_xyzrgb(
        args.out_dir / "corrupted.ply",
        xyz,
        jnp.asarray(np.rint(corrupted_255), dtype=jnp.uint8),
    )
    save_ply_xyzrgb(
        args.out_dir / "recon.ply",
        xyz,
        jnp.asarray(np.rint(np.clip(predicted_255, 0, 255)), dtype=jnp.uint8),
    )
    print(f"outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
