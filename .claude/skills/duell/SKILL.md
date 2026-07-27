---
name: duell
description: Set up and run a timed duel in which the user and an autonomous agent independently try to beat a metric on the same task, blind to each other, then consolidate what both learned. Use when the user says "Duell", "duel", "lass uns gegeneinander antreten", "wer schafft die bessere Success Rate", "ich fordere den Agent heraus", or wants to race an agent on an experiment under fixed conditions. Covers the goal interview, the rules document, the frozen zone and its verification gate, the baseline run, the start prompt, and the post-duel consolidation.
---

# Duell

A duel is a timed, blind competition: the user and an agent independently try
to improve the same metric on the same task, under identical frozen
conditions. Afterwards both sides exchange what they tried, so the exercise
produces knowledge rather than just a winner.

The value is in the conditions being airtight and machine-checkable. A duel
whose rules cannot be verified from a git diff is a duel that measures nothing.

## Phase 1 - define goal and conditions

**Invoke the `grill-with-batch` skill.** Do not guess at the setup; interview
until the frontier is empty.

Dispatch sub-agents for the facts before the first round, in parallel:

- What the target task actually is - config, entry point, metric, seeds
- The compute reality - partitions, walltime, queue behaviour, throughput
- The repo's own rules and automation - PR workflow, worktrees, review gates

Decisions the tree must reach. Each of these has bitten a real duel setup:

| Decision | Why it matters |
|---|---|
| Total time, and what counts into it | Queue time is usually the dominant cost |
| What is optimized | Training sprint, eval tuning, or code-only |
| Which arm or variant | Throughput differences can decide the duel by themselves |
| The baseline, and who runs it | An outdated or disowned number is not a baseline |
| The frozen zone, as **paths** | Anything else cannot be verified |
| The comparison point | Equal walltime and equal steps are different duels |
| Secondary criterion and its tie-break threshold | |
| Where every side writes its notes | |
| Blind or open | |
| Parallel job limit per side | |

**Throughput trap.** If the two sides run encoders with different per-step
cost, equal walltime measures encoder cost, not idea quality. Either compare
at **equal step count**, or say explicitly that speed is part of what is being
judged.

**Signal trap.** Compute what the time budget actually buys in training steps,
subtract prefill, and tell the user whether the remaining steps can produce a
distinguishable metric at all. If they cannot, say so plainly and let the user
decide. A short first duel that mainly tests the format is a legitimate goal -
it just must not be mistaken for an experimental result.

## Phase 2 - write the duel folder

One folder under `prototyp/<duell-name>/`, following `prototyp/CLAUDE.md`:

```
prototyp/<duell-name>/
  GOAL.md        goal, format, success criteria, measurement protocol
  RULES.md       time, frozen zone, free ground, compute, branch and PR format
  PLAN.md        orchestrator role, subagents, workflow, candidate list
  PROBLEMS.md    known blockers, verified from the code, with file:line
  verify.sh      the gate
  HANDOFF.md     index over the dated run folders
  <YYYY-MM-DD>/
    README.md    layout and writing rules for the agents
    LEDGER.md    the result ledger, maintained by the orchestrator only
    agents/<role>/NOTES.md
    runs/        logs, copied metrics files, rendered job scripts
```

Writing rules that go into the dated README:

- Each agent writes **only** into its own folder; other folders are read-only.
- Reading each other is the point - it is how a subagent learns an idea is
  already dead without the orchestrator relaying it.
- The ledger is the orchestrator's alone.
- Every number carries a source: job id, run directory, tracking id, or
  `file.py:line`.
- Dead ends are worth as much as successes.

Everything the duel produces stays in this folder. Do not scatter it into the
repo's documentation tree; that is for what survives the duel.

## Phase 3 - the frozen zone and its gate

State the frozen zone as **paths, not intentions**. "Do not change the
environment" is unenforceable; a path list is checkable in one command.

Include every file that could change the measured number, not only the obvious
ones. A metric aggregator living outside the environment package still decides
what the metric says.

`verify.sh` checks three things:

1. `git diff --name-only <base>...HEAD -- <frozen paths>` is empty, **and** so
   is `git status --porcelain` over the same paths.
2. Checksums of frozen data files that are gitignored, recorded once before the
   duel with a `--record` mode.
3. The frozen seed appears in every changed or added job config.

Run it before every PR and at the end of the duel. A failing gate invalidates
the run.

## Phase 4 - baseline and start

The baseline is run **before** the clock starts, on the main branch, so both
sides can read its metrics file. It uses the same command shape a graded run
uses - not a shortened smoke variant, whose step caps and environment
differences make it a different experiment.

Record in the ledger: run directory, job id, seed, reached steps, the metric,
episode count, derived time per step, peak memory.

Then hand the agent a start prompt of this shape:

```
Du trittst in einem Duell gegen <Name> an. Lies zuerst und vollstaendig:
  prototyp/<duell-name>/GOAL.md
  prototyp/<duell-name>/RULES.md
  prototyp/<duell-name>/PLAN.md
  prototyp/<duell-name>/PROBLEMS.md

Ab deinem ersten Tool-Call laeuft eine Uhr von <N> Stunden Gesamtzeit.
Alles zaehlt: Nachdenken, Code, Queue, Laufzeit, Auswertung, PR.

Ziel: <Zielsatz in einem Satz>.

Du bist Orchestrator, nicht Einzelarbeiter. Delegiere nach PLAN.md, halte
deinen eigenen Kontext schlank. Schreibe fortlaufend in
prototyp/<duell-name>/<datum>/ nach den dort dokumentierten Regeln.

Baseline: <Pfad, Zahlen, wie der Vergleichspunkt abzulesen ist>

Fang an.
```

Run the agent as an orchestrator at high effort with parallel subagents. The
bottleneck in a timed duel is queue and runtime, not thinking - so the agent
should submit its first candidate early and keep working while it waits.

If personas or specialist subagents should advise the agent, place them
somewhere the agent can reach that is **not** the repository, especially when
their source material contains anything private.

## Phase 5 - consolidation

When the clock runs out, both sides bring their ledgers. Merge into one table:
what each side tried, what it measured, what held up. Write the joint verdict
into the same dated ledger.

Only then decide what graduates out of the duel folder into the repo's own
documentation, and what becomes a follow-up task.

## Things learned the hard way

- Facts kill plans. Expect several of your own recommendations to die during
  the interview and again during the baseline run. Correct them in one line
  and continue.
- A failed job is not automatically a bug in the change. Give the agent an
  explicit rule for distinguishing infrastructure noise from real breakage,
  with a retry budget, so it does not spend a quarter of the duel debugging a
  transient abort.
- Check that the baseline run actually did what it claims. A run can log
  episodes for half an hour while silently never logging a single training
  step.
- Prefer a metric the harness already writes to disk over one you have to add.
  Verify it is present in a real run before making it a criterion.
