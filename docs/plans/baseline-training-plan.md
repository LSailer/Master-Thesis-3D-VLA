# R2Dreamer No-Goal Baseline Training Plan

**Branch:** `feature/baseline-training`
**Date:** 2026-04-10
**Status:** Planned

## Goal

Validate that the R2Dreamer world model learns Habitat ObjectNav dynamics and outperforms a random agent. This is the foundational baseline before adding goal conditioning or 3D features.

## Success Criteria

The no-goal R2Dreamer baseline is validated when **all four** hold:

1. **World model learns** -- dyn/rep losses decrease and stabilize (RSSM models Habitat dynamics)
2. **Agent explores** -- action distribution doesn't collapse; episode lengths vary (not all hitting max 500)
3. **Reward improves** -- mean episode reward trends upward (agent learns to reduce geodesic distance)
4. **Qualitative** -- top-down trajectory plots show purposeful movement, not random spinning

**Comparison:** A random-action agent is run on the same 50 episodes. The trained agent must clearly outperform random on reward trend and trajectory quality.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Episode filter threshold | <200 shortest-path steps | Removes ~6.5% broken/pathological episodes (all at 500 steps), not difficulty filtering |
| Filter implementation | Pre-compute and cache | Avoids re-computing on every SLURM job; deterministic across runs |
| Training duration | 2.4M steps | Sufficient for learning signal; extend to 5M only if reward still trending up |
| Eval episodes | 50 | Enough for confidence intervals, ~25 min runtime |
| WandB additions | Action distribution + latent diagnostics | Highest signal for diagnosing "is agent exploring?" and "is world model learning?" |
| Semantic overlay | Included | Needed to evaluate whether navigation is semantically meaningful |
| Script refactoring (#52) | Deferred | Tech debt, doesn't block training or produce thesis results |
| Periodic eval (#51) | Closed | Already implemented in `run_jax_habitat.py` |

## Deliverables

### Phase 1 -- Preprocessing

**1. Precompute episode step counts**
- New script: `modules/envs/scripts/precompute_episode_steps.py`
- Runs `ShortestPathFollower` on all train + val episodes
- Saves to `data/episode_step_counts.json` (checked into git)
- Format: `{split: {episode_id: shortest_path_steps}}`
- One-time SLURM job (~20-60 min)

**2. Episode filtering in environment**
- Modify: `modules/envs/habitat.py` (`HabitatObjectNavEnv.__init__`)
- Read `data/episode_step_counts.json` at init
- Filter out episodes with >= 200 shortest-path steps
- Log how many episodes were filtered

### Phase 2 -- Logging Improvements

**3. Action distribution logging**
- Modify: `modules/r2dreamer/scripts/run_jax_habitat.py`
- Log per-episode action percentages to WandB:
  - `action/stop_pct`, `action/forward_pct`, `action/left_pct`, `action/right_pct`
- Detects action collapse (e.g., always turning left)

**4. Latent diagnostics logging**
- Modify: `modules/r2dreamer/agent.py` (train_step return dict)
- Add to training metrics:
  - `latent/prior_entropy`, `latent/posterior_entropy`, `latent/kl_divergence`
- Shows whether RSSM is learning meaningful dynamics vs outputting uniform noise

### Phase 3 -- Evaluation Pipeline

**5. Trajectory saving in eval**
- Modify: `modules/r2dreamer/scripts/eval_habitat.py`
- Always record per step: `agent_position (x, y, z)`, `agent_heading`
- Always record per episode: `start_position`, `goal_position`
- No flag needed -- data is tiny, always useful

**6. Random baseline script**
- New script: `modules/r2dreamer/scripts/eval_random_habitat.py`
- Runs 50 episodes with random actions
- Same output JSON format as `eval_habitat.py` (positions, reward, success, actions)
- Follows notebook workflow: script computes and saves to `output/`

**7. Evaluation notebook -- top-down semantic maps**
- New/updated notebook: `modules/r2dreamer/notebooks/baseline_evaluation.ipynb`
- Top-down map rendering:
  - Navigable area from navmesh
  - Semantic overlay from `.basis.glb` scene files (colored by object category)
  - Agent trajectory as colored path
  - Start position (green marker)
  - Stop position (red marker)
  - Goal object position (gold marker)
- Per-episode and aggregate views

**8. Comparison notebook -- random vs trained**
- New notebook: `modules/r2dreamer/notebooks/baseline_comparison.ipynb`
- Side-by-side comparison on all 4 success criteria:
  - Loss curves (trained only)
  - Action distribution histograms (random vs trained)
  - Reward distributions (random vs trained)
  - Top-down trajectory maps (random vs trained, same episodes)

### Phase 4 -- Training

**9. Updated SLURM script**
- Modify: `modules/r2dreamer/scripts/slurm/train_habitat_baseline.sbatch`
- 2.4M steps with episode filtering enabled
- Eval every 50k steps, 50 episodes on val split
- New WandB tags reflecting this run

## Implementation Order

```
Phase 1 (preprocessing)     Phase 2 (logging)       Phase 3 (eval)          Phase 4 (train)
  [1] precompute script  -->  [3] action dist    -->  [5] trajectory save -->  [9] SLURM submit
  [2] env filtering      -->  [4] latent diags   -->  [6] random baseline
                                                  -->  [7] eval notebook
                                                  -->  [8] comparison notebook
```

Phases 1-3 are implemented and tested before submitting the training job in Phase 4.

## Related Issues

- #49 -- Preprocess data: filter episodes with >200 shortest path steps (addressed by deliverables 1-2)
- #51 -- Add periodic evaluation to Habitat training loop (closed, already implemented)
- #52 -- Refactor: extract shared boilerplate from training scripts (deferred)

## Out of Scope

- Script refactoring (#52)
- Goal-conditioned R2Dreamer (future experiment after baseline validated)
- UNITE 3D feature integration (requires validated baseline first)
