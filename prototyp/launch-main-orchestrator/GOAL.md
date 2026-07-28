# Goal: main as the orchestrator

Refactor the launch path so `src/main.py` owns the run loop instead of
delegating it through three layers (`train()` -> `run_training` -> `train_loop`).
Structure and semantics change together in one pass (no bit-identity goal);
verification is an A/B SLURM run plus the render-and-parse test.

## Target shape

```python
def main(argv=None):
    args = build_parser().parse_args(argv)          # ONE parser, src/launch/parser.py
    adapter_cls = resolve_adapter(args.adapter)
    env = make_env(args.env, curriculum=args.curriculum, ...)
    adapter = make_adapter(adapter_cls, args, output_dir=args.output_dir)
    agent = ...                                      # eval: params from --checkpoint
    collector = ExperienceCollector(...)             # both modes, eval: buffer=None
    logger = RunLogger(...)

    if args.overfit_one_batch:
        return overfit(...)                          # own branch, never enters the loop

    carry = agent.initial_act_state()
    train_credit = 0.0
    with run_session(logger, collector, env, hard_exit=...):   # teardown CM
        for step in range(args.steps):               # env-step is the loop unit
            if args.mode == "train" and step < args.prefill:
                action = uniform_random()            # prefill stays a distinct phase
            else:
                action, carry = inference(agent, params, obs, carry, ...)
            ...env step via collector, int(action) happens here...
            if args.mode == "train":
                train_credit += train_ratio / batch_steps
                while train_credit >= 1.0:           # gate visible in main
                    train(agent, batch, ...)         # exactly one gradient step
            if args.mode == "eval" and episodes_done == args.episodes:
                break
```

- `inference` = exactly one env step, functional carry threaded by main.
- `evaluate` as a separate loop owner disappears; `--mode eval` is the same
  loop without the train call, params loaded from `--checkpoint`.
- Eval artifacts (eval_results.json, topdown PNGs, W&B videos) stay, but main
  writes them at episode end; episode cap is `--max_episode_steps` (default 500).
- Validation (`val_loop`, val flags, val YAML keys) is deleted entirely.
- `scripts/r2dreamer/run.py` + `_run_configs.py` are deleted; production goes
  through `python -m src.main` (YAML: `script: -m src.main`, renderer unchanged).

## Why

- Today `main()` has zero production traffic (SLURM path bypasses it entirely)
  and the loop ownership is split across three files.
- One parser removes the parse_known_args remainder contract, the triple
  `--curriculum` definition, and the train/eval default divergence.
- One functional act path removes the snapshot/restore hack and four
  isinstance(RandomAgent) branches.
