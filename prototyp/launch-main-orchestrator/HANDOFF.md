# Handoff state

## 2026-07-28 - interview complete, implementation not started

- Design interview (batch-grill-me) ran over 5 rounds; ALL 43 decisions are
  settled in DECISIONS.md. Frontier empty, shared understanding confirmed.
  Waiting for the explicit go before any implementation.
- No code has been touched. Worktree: launch-main-orchestrator-599c0d,
  branch claude/launch-main-orchestrator-599c0d.
- Ground-truth maps gathered by exploration (structure + consumers) are
  reflected in DECISIONS.md/PROBLEMS.md; key facts:
  - Today main() has zero production traffic; SLURM path is
    launch.sh -> launch.py -> run.py <run-id> -> _run_configs.launch_run ->
    src.main.train(**cfg, argv=flags).
  - The real loop is src/r2dreamer/launch/loops.py:train_loop (env-step loop,
    train_credit gate at :452).
  - Act paths: stateful agent.act (train/val, mutates _act_state) vs
    functional act_with_state (eval). Jit feasibility of a single public
    jitted act confirmed; params must be explicit (stale-params hazard).
- Implementation plan written: PLAN.md (3 PRs, tasks T1-T4, wave order).
- Wave 1 COMPLETE 2026-07-28 evening: PR #219 (launcher bools, Scalar-typed)
  and PR #220 (agent API, pure jitted act/train_step) open, both green, not
  merged. Process rules now standing: no no-mistakes pipeline, no Any types,
  pyright+pylint in the worktree BEFORE commit, test validation via srun on
  dev_cpu,cpu_il,cpu. Gate-CPU rebuild: agent stopped by owner, worktree
  deleted, work preserved on branch worktree-agent-ad8a78a55c7a88c1e
  (ee04059); owner rebuilds the gate manually.
- Next: T3 implemented directly by the firstmate in THIS worktree (owner
  speed decision pending final go): merge PR #220 branch in, then parser ->
  orchestrator -> YAML migration as clean commits, one srun validation, PR3
  stacked on #220. Carry-over list for PR3 is in PROBLEMS.md.
  Next step on go: spawn wave 1 (T1 agent refactor + T2 launcher booleans,
  parallel shipmates), then T3a-c sequential in this worktree, then T4
  verification runs (l1_cnn + hybrid, ~5k steps, seed 42).
