# When to schedule gradient/train steps relative to env steps in Dreamer-style MBRL

Research findings from primary sources (papers + official repos). Compiled 2026-07-23.

---

## TL;DR

- **DreamerV3 — the algorithm you are cloning — does NOT defer to episode boundaries.** It interleaves training with data collection at a *fixed replay ratio*, applying a *fractional-credit accumulator* (`elements.when.Ratio`) after every policy step that returns 0, 1, or several `train` calls. This is exactly the "credit accounting" you already implement. Episode-boundary batching is *not* "the DreamerV3 standard."
- **Episode-boundary deferral was only the DreamerV1 *paper* pseudocode** (Algorithm 1: run `C=100` updates, then collect *one* full episode with frozen params). Even DreamerV1's *released code* interleaves via a `should_train` step-counter, and DreamerV2 interleaves every `train_every` env steps via a per-step callback. So "collect-then-update" is a presentational framing, not what the reference implementations do.
- **The replay-ratio / UTD literature parameterizes learning by the *ratio* of gradient steps to env steps, never by the intra-episode *timing* of updates** (Fedus 2020, REDQ 2021, D'Oro 2023, Nikishin 2022). At a fixed ratio, per-step vs. episode-boundary is not a variable any of these papers identifies as changing learning dynamics — the levers are ratio, data staleness/age, and plasticity loss.
- **Within-episode policy staleness is benign for Dreamer-style MBRL.** The world model is trained by *supervised* learning on replayed *off-policy* sequences and the actor is trained *entirely inside imagination*, so there is no importance-sampling correction on real-env data (unlike IMPALA V-trace / R2D2). DayDreamer runs actor and learner as fully asynchronous threads with **no ratio limiter and no off-policy correction** and still learns on real robots in <1 h.
- **Net practical answer:** at a fixed ratio, episode-boundary deferral does *not* reduce wall-clock and does *not* improve learning; it only makes updates *bursty* (up to `500·R` updates dumped at once for your 500-step episodes) and *raises the average age of the newest online data* the model sees. A two-thread actor/learner overlap can cut wall-clock *only if* env-stepping and training occupy separable hardware/critical paths; it converts your fixed ratio into an emergent one and is theoretically safe for Dreamer.

---

## Sub-question 1 — What schedule do DreamerV1/V2/V3 papers AND official repos actually use?

### DreamerV1 — "Dream to Control" (Hafner et al., arXiv:1912.01603)

**Paper Algorithm 1 is episode-boundary (collect-then-update), verbatim** (arXiv:1912.01603, p. 3, Algorithm 1):

```
Initialize dataset D with S random seed episodes.
while not converged do
  for update step c = 1..C do
    // Dynamics learning   ... update θ
    // Behavior learning   ... update φ, ψ
  // Environment interaction
  o1 ← env.reset()
  for time step t = 1..T do
    Compute s_t ~ p_θ(...);  a_t ~ q_φ(a_t|s_t);  add exploration noise
    r_t, o_{t+1} ← env.step(a_t)
  Add experience to dataset D
```

So the pseudocode: run `C` update steps, **then** roll out one full episode of `T` steps under *frozen* params, then repeat. Hyperparameters named: seed episodes `S`, **collect interval `C`**, batch size `B`, sequence length `L`, imagination horizon `H` (arXiv:1912.01603, Algorithm 1 sidebar).

The appendix pins the numbers (arXiv:1912.01603, "Environment interaction"): *"The dataset is initialized with S = 5 episodes… We iterate between **100 training steps and collecting 1 episode** by executing the predicted mode action with Normal(0, 0.3) exploration noise… we fix [action repeat] to 2."* → On DMC (1000 raw steps / repeat 2 = 500 agent steps/episode) this is **~100 grad steps per ~500 agent steps ≈ ratio 0.2 grad/step**, applied as one burst per episode.

**But the released code interleaves.** `danijar/dreamer/dreamer.py` (`main()` loop) collects for `config.eval_every` steps at a time via `tools.simulate(agent, train_envs, steps, …)`; inside the `Dreamer.__call__`, training is gated by a step-counter, not an episode boundary:

```python
if self._should_train(step):
    n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
    for train_step in range(n):
        self.train(next(self._dataset), log_images)
```

i.e. every `train_every` env steps run `train_steps` gradient iterations (`danijar/dreamer@master dreamer.py`). At DMC defaults this reproduces the "~100 updates per episode" of Algorithm 1, but the *mechanism is per-step interleaving*, not a hard episode boundary. **Takeaway: the "collect-then-update" picture is the paper's exposition, not a hard requirement of the code.**

### DreamerV2 — "Mastering Atari with Discrete World Models" (arXiv:2010.02193, `danijar/dreamerv2`)

Interleaves via a per-env-step callback. `danijar/dreamerv2/dreamerv2/train.py`:

```python
should_train = common.Every(config.train_every)
def train_step(tran, worker):
    if should_train(step):
        for _ in range(config.train_steps):
            mets = train_agent(next(train_dataset))
train_driver.on_step(train_step)      # ← registered as a PER-STEP callback
```

`common.Every(train_every)` fires every `train_every` env steps; it is attached to the driver's `on_step`, so training is checked **after every environment step**, never at episode boundaries. Config defaults (`danijar/dreamerv2/dreamerv2/configs.yaml`): `train_every: 5`, `train_steps: 1`, `pretrain: 1`, `batch (size): 16`, `length: 50`, `prefill: 10000`; **Atari override `train_every: 16`, `prefill: 50000`**; DMC overrides `pretrain: 100`, `prefill: 1000`. So DreamerV2 defaults to **1 grad step every 5 env steps** (Atari: every 16).

### DreamerV3 — "Mastering Diverse Domains through World Models" (arXiv:2301.04104 / Nature 2025, `danijar/dreamerv3`)

**Parameterized by a fixed replay ratio, applied per-step via a fractional accumulator — the canonical answer to your question.**

Definition (Nature 2025 version, "Implementation → Experience replay"): *"We parameterize the amount of training via the **replay ratio**. This is the fraction of time steps trained on per time step collected from the environment, without action repeat. Dividing the replay ratio by the time steps in a minibatch and by action repeat yields the **ratio of gradient steps to env steps**. For example, a replay ratio of 32 on Atari with action repeat of 4 and batch shape 16×64 corresponds to **1 gradient step every 128 env steps**, or 1.5M gradient steps over 200M env steps."*

Replay buffer is a *uniform buffer with an online queue*: *"each minibatch is formed first from non-overlapping online trajectories and then filled up with uniformly sampled trajectories from the replay buffer"* (same section). This means the newest data is *guaranteed* into each batch — burstiness at episode boundaries would change what "newest" means.

**The scheduling loop** (`danijar/dreamerv3/embodied/run/train.py`):

```python
should_train = elements.when.Ratio(args.train_ratio / batch_steps)   # batch_steps = batch_size*batch_length
...
for _ in range(should_train(step)):
    ... agent.train(carry_train[0], batch)
```

`should_train(step)` is evaluated in the driver's per-policy-step path, and returns *how many* train calls to run this step (0, 1, or more). **The fractional-credit accumulator you described is literally this class** (`danijar/elements@main elements/when.py:26-43`):

```python
class Ratio:
  def __init__(self, ratio):
    self._ratio = ratio
    self._prev = None
  def __call__(self, step):
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._ratio < 0:
      return 1
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)   # how many updates this env step
    self._prev += repeats / self._ratio                 # carry the fractional remainder
    return repeats
```

Line `self._prev += repeats / self._ratio` keeps the sub-step remainder so that over time the realized update count converges to exactly `ratio · (steps collected)` — identical accounting to `train_ratio/(batch·seq_len)` credit per env step. **This is per-step interleaving, not episode-boundary batching.**

Per-benchmark `train_ratio` values (`danijar/dreamerv3/dreamerv3/configs.yaml`; default `train_ratio: 32.0`, `batch_size: 16`, `batch_length: 64`):

| Benchmark | `train_ratio` |
|---|---|
| default / atari / dmlab | 32 |
| procgen | 64 |
| dmc_vision, loconav | 256 |
| atari100k | 256 |
| crafter | 512 |
| dmc_proprio, bsuite | 1024 |

(These are "replayed steps per env step"; divide by `batch_size·batch_length = 1024` and by action-repeat to get grad-steps-per-env-step — e.g. dmc_proprio 1024 → ~1 grad step/env step before action repeat.)

The Nature version also reports (Results / Fig. 6): *"Higher replay ratios predictably increase the [data efficiency]… The replay ratio affects the number of gradient updates performed by the agent… robust learning with fixed hyperparameters across the compared model sizes and replay ratios"* — i.e. the tunable is the *ratio*, chosen "to fit the step budget of each benchmark," never the intra-episode schedule.

**DreamerV3 also ships an async mode** (`embodied/run/parallel.py`, invoked when the acting and learning are split across processes), the same pattern DayDreamer uses; the default single-process `run/train.py` above is the synchronous interleaved one.

---

## Sub-question 2 — Replay-ratio / UTD literature: does *granularity* matter, or only the *ratio*?

**Consensus of the primary sources: the controlled variable is the RATIO (updates per env step) and the resulting data *age/staleness* — not the fine-grained *timing* of when within an episode the updates land.**

- **Fedus et al., "Revisiting Fundamentals of Experience Replay," ICML 2020 (arXiv:2007.06700).** They isolate exactly two properties: *"the replay capacity and the ratio of learning updates to experience collected (replay ratio)."* Method: they *directly control the replay ratio* and vary capacity independently, and find *"uncorrected n-step returns are uniquely beneficial"* for tolerating older data. The paper's entire framing is ratio + capacity + data-age (the "oldest policy" in the buffer); **update scheduling granularity is not a factor they vary.** (Abstract; §"replay ratio" definition.)

- **REDQ, Chen et al., ICLR 2021 (arXiv:2101.05982).** Defines UTD directly (§3): *"the UTD ratio… is **the number of updates taken by the agent compared to the number of actual interactions with the environment**."* They use **G = 20** updates per env interaction (arXiv:2101.05982 §"REDQ", *"we find G = 20 work[s] well"*; MBPO uses 20–40; SAC uses G = 1). The high UTD is what forces their ensemble + in-target minimization to control Q-bias. **Again the lever is the count/ratio, not the schedule.**

- **D'Oro et al., "Sample-Efficient RL by Breaking the Replay Ratio Barrier," ICLR 2023 (OpenReview `0pC-9aBBVJe`; dblp DOroSNBBC23).** Replay ratio = number of gradient updates per environment step; their reference implementation exposes it as `--updates_per_step` (e.g. **32**) (`proceduralia/high_replay_ratio_continuous_control`, README). Core claim: *fully/partially **resetting** agent parameters lets you scale the replay ratio by orders of magnitude for a fixed interaction budget* — i.e. the barrier is about *how high the ratio can go* (plasticity loss), **not about the intra-episode timing.**

- **Nikishin et al., "The Primacy Bias in Deep RL," ICML 2022 (arXiv:2205.07802).** Complementary mechanism: high update counts on early data cause overfitting to early experience; the fix is *periodic resets*, not rescheduling. Reinforces that the failure mode at high ratio is *plasticity/primacy*, a function of ratio and data age — **schedule granularity is not implicated.**

**Interpretation for your case:** at a *fixed* ratio, moving from per-step interleaving to episode-boundary batching keeps the same total updates and the same UTD, so none of these papers predicts a first-order change in learning. The only knobs these papers say matter — ratio value, capacity, data age, plasticity — are affected by episode-boundary deferral *only second-order*, via (a) burstiness of updates and (b) the average staleness of the newest online-queue data (deferral lets up to a full episode of fresh transitions sit un-trained-on before the burst).

---

## Sub-question 3 — Async actor/learner precedents & within-episode policy staleness

**Two distinct staleness regimes. Dreamer-style MBRL sits in the benign one.**

### DayDreamer (Wu, Escontrela, Hafner et al., 2022, arXiv:2206.14176) — the Dreamer-native async design

- **Decoupled threads:** *"We decouple learning updates from data collection… a **learner thread** continuously trains the world model and actor critic behavior, while an **actor thread** in parallel computes actions for environment interaction."* (arXiv:2206.14176 §2, "Robot Learning").
- **No ratio limiter at all:** *"Compared to Hafner et al. (2020), **there is no training frequency hyperparameter** because the decoupled learner optimizes the neural networks in parallel with data collection, **without rate limiting**."* (arXiv:2206.14176 §2, end of Actor-Critic subsection.) → In async mode the ratio becomes *emergent* (whatever throughput the learner achieves), not a set value.
- **Why staleness is tolerated:** *"The world model is trained on replayed **off-policy** sequences through **supervised** learning."* (arXiv:2206.14176, Fig. 2 caption / §2.) The actor is trained *inside imagined latent rollouts* ("massively parallel behavior learning… without decoding observations"), so real-environment actions never enter a policy-gradient term that needs importance weighting. The acting thread simply uses the most recently synced parameters; a stale actor only shifts the *data distribution/exploration*, not the correctness of any gradient.
- **Motivation is latency/throughput:** *"We develop an asynchronous actor and learner setup, which is essential in environments with high control rates, such as the quadruped, and also accelerates learning for slower environments."* (arXiv:2206.14176 §3, "Implementation".) It is built on the *official DreamerV2 implementation*.

### On-policy distributed methods that DO need correction (contrast class)

- **IMPALA (Espeholt et al. 2018, arXiv:1802.01561):** decoupled actors + central learner; because *"actors inevitably operate with outdated policy parameters compared to the continuously-updating learner,"* it introduces **V-trace off-policy correction** (truncated importance sampling) to fix the actor–learner policy lag. Correction is needed *because IMPALA's loss is a policy-gradient/critic loss on real-env trajectories.*
- **R2D2 (Kapturowski et al., ICLR 2019, OpenReview `r1lyTjAqYX`):** distributed prioritized replay (Ape-X-style) with RNNs. Studies *"the effects of **parameter lag** resulting in **representational drift and recurrent state staleness**,"* and mitigates with **(1) storing the recurrent state** in replay and **(2) a burn-in** prefix so the RNN re-warms its hidden state before the loss is applied. This staleness problem is specific to *value-based* recurrent replay, not to Dreamer.
- **Ape-X / SEED RL:** same family — actors lag the learner; Ape-X tolerates it with prioritized replay + n-step (off-policy value learning), SEED RL centralizes inference to *reduce* lag. All are value/policy-gradient methods on real data.

**Why Dreamer needs no such correction:** its two learning signals are (i) *self-supervised* world-model reconstruction/dynamics on replayed sequences (correct for any data source — off-policy by construction) and (ii) actor-critic learning *on model-imagined rollouts* generated from the *current* params (on-policy *in the model*, regardless of which behavior policy collected the seed states). There is no term of the form "policy-gradient on real environment actions" that would require V-trace/importance weights. Hence within-episode acting-policy staleness (whether a few steps in interleaved mode, or a whole 500-step episode in deferred mode, or continuous drift in async mode) is *not* a source of bias for the learning update — it only changes exploration and the composition of the replay buffer.

---

## Sub-question 4 — Habitat ObjectNav-specific evidence

- **DD-PPO (Wijmans et al., ICLR 2020, arXiv:1911.00357)** — the standard Habitat nav baseline — is explicitly **synchronous & on-policy**: *"DD-PPO is distributed (uses multiple machines), decentralized (lacks a centralized server), and **synchronous (no computation is ever stale)**."* Every PPO update consumes *freshly collected on-policy rollouts*; workers synchronize gradients each rollout. **Implication:** with an on-policy method, updates are *necessarily* batched at rollout boundaries and the behavior policy is fixed within a rollout — so "episode/rollout-boundary updates with frozen params" is *forced* by on-policy RL, and does *not* generalize to off-policy Dreamer, where it is a free choice. Do not import DD-PPO's rollout-boundary structure as evidence that Dreamer "should" defer to episode boundaries; the two have different reasons.
- **World models on ObjectNav do exist but are not primary-source-detailed on scheduling.** A DreamerV3-based zero-shot ObjectNav method on **Habitat 0.3.1, ObjectNav-HM3D v1** is reported ("A DreamerV3 Framework for Sample-Efficient 3D Zero-Shot Goal Navigation," ICBDAIRM 2025, dl.acm.org/10.1145/3800227.3800282 — paywalled, schedule not extractable) and **WMNav** (arXiv:2503.02247) integrates a VLM into a world model for object-goal navigation. **No primary source found that ablates train-step *scheduling* (interleave vs. defer vs. async) specifically on Habitat ObjectNav** — see Open Questions.

---

## Implications for this thesis

Your current design (DreamerV3-style, JAX, fixed `train_ratio`, per-env-step credit `train_ratio/(batch·seq_len)`) **is already the DreamerV3 reference behavior** — it is a re-implementation of `elements.when.Ratio` + `embodied/run/train.py`. You do not need to change anything to be "Dreamer-standard"; you already are.

**(a) Deferring accrued updates to episode boundaries (500-step episodes):**
- *Learning dynamics:* At the same ratio, total updates and UTD are unchanged, so the replay-ratio literature (Fedus, REDQ, D'Oro, Nikishin) predicts **no first-order change**. Two *second-order* effects, both mild-to-adverse for you:
  1. **Burstiness.** You would dump up to `500 · (train_ratio/(batch·seq_len))` updates in a single burst at each episode end, then zero during the next episode. That concentrates optimizer noise and, with DreamerV3's *online-queue* batching, means the "newest online trajectories" injected into each batch all originate from one just-finished episode (one behavior policy) rather than a rolling mix. Interleaving keeps the online queue fresher and the update stream smoother.
  2. **Increased newest-data staleness.** Under deferral, a transition collected at step 1 of a 500-step episode waits ~500 env steps before *any* gradient sees it; under interleaving it can be trained on within a few steps. Fedus explicitly flags *data age* as the thing that matters (and that uncorrected n-step returns are needed to tolerate it). Net: deferral moves you in the *wrong* direction on the one axis the literature says is load-bearing.
- *Wall-clock:* **No saving.** Same number of `train_step`s and same number of `env.step`s; a single-thread agent serializes them either way. Deferral only reshapes *when* the compute happens (latency spikes of up to 500·R updates at episode ends), which can also complicate SLURM step-time accounting (cf. your ~219 ms/step budget — a burst would show as a periodic stall, not a lower mean).
- *Verdict:* Episode-boundary deferral is **not** "the Dreamer standard" (that belief comes from the DreamerV1 *paper pseudocode*; V2/V3 code and even V1 code interleave). It buys nothing at fixed ratio and slightly worsens data freshness/burstiness. **Keep per-step interleaving.**

**(b) Two-thread actor/learner overlap:**
- *Correctness:* **Safe for Dreamer** — DayDreamer runs exactly this with *no ratio limiter and no off-policy correction* and learns on real robots (arXiv:2206.14176). You do *not* need V-trace/burn-in-style corrections (those are for on-policy/value-based real-data losses — IMPALA, R2D2); Dreamer's model loss is supervised-off-policy and its actor loss is in-imagination.
- *What changes:* The ratio becomes **emergent** (learner throughput ÷ actor throughput) instead of a set value — you lose the exact `train_ratio` knob unless you add rate-limiting back. Within-episode param staleness (actor uses last-synced weights) is benign for the gradient but does change exploration/data distribution.
- *Wall-clock:* Overlap helps **only if** env-stepping and `train_step` occupy separable critical paths — e.g. Habitat env stepping on CPU while `train_step` runs on GPU, so they run concurrently instead of serially. Given your profile (VGGT 132 ms + replay_sample 59 ms per step), an async split could hide the CPU-side env/replay cost behind GPU training *iff* the two don't already contend for the same device. This is an engineering win, not a learning-quality change.
- *Verdict:* Worth prototyping for **throughput**, not for learning quality. If you adopt it, either accept an emergent ratio or re-introduce a rate limiter to hold `train_ratio` fixed for controlled comparisons.

**One-line recommendation:** stay with per-step interleaved fixed-ratio training (you already match DreamerV3); consider async only as a wall-clock optimization when env/replay CPU work can overlap GPU training; do **not** switch to episode-boundary deferral.

---

## Annotated reading list (ranked for the discussion section)

1. **DreamerV3 — Hafner et al., "Mastering Diverse Domains through World Models," arXiv:2301.04104 / Nature 2025.** *Primary anchor.* The "Experience replay" implementation paragraph defines the replay ratio and its grad-steps-per-env-step conversion; use it to justify per-step interleaving as canonical. Pair with the repo (`configs.yaml`, `embodied/run/train.py`, `elements/when.py:26-43`).
2. **`danijar/dreamerv3` + `danijar/elements` source.** *Ground truth for the schedule.* `elements/when.py` `Ratio` class is literally your credit accountant; cite the file/lines to show your implementation == reference.
3. **DayDreamer — Wu, Escontrela, Hafner et al., arXiv:2206.14176.** *Primary anchor for async + staleness.* "No training frequency hyperparameter… without rate limiting," and the off-policy-supervised model-learning justification. Use for the two-thread option and to argue staleness is benign.
4. **Fedus et al., "Revisiting Fundamentals of Experience Replay," ICML 2020, arXiv:2007.06700.** *Ratio-vs-capacity-vs-age framing.* Best citation that "ratio and data age matter, timing granularity is not a studied lever," and that n-step tolerates staleness.
5. **DreamerV1 — Hafner et al., "Dream to Control," arXiv:1912.01603.** *For the historical claim.* Algorithm 1 is the *only* place "collect-then-update / episode-boundary" appears; use it to explain where the user's belief comes from and to contrast with the released code.
6. **D'Oro et al., "Breaking the Replay Ratio Barrier," ICLR 2023 (OpenReview 0pC-9aBBVJe).** *High-ratio scaling & resets.* Cite for "the barrier is how high the ratio can go (plasticity), not the schedule."
7. **REDQ — Chen et al., ICLR 2021, arXiv:2101.05982.** *UTD definition + G=20.* Clean definition of update-to-data ratio as the controlled quantity.
8. **Nikishin et al., "The Primacy Bias in Deep RL," ICML 2022, arXiv:2205.07802.** *Failure mode at high ratio.* Resets, not rescheduling, are the fix — supports "granularity isn't the lever."
9. **DD-PPO — Wijmans et al., ICLR 2020, arXiv:1911.00357.** *Habitat baseline contrast.* "Synchronous — no computation is ever stale"; explains why on-policy nav *forces* rollout-boundary updates and why that doesn't transfer to Dreamer.
10. **IMPALA (arXiv:1802.01561) & R2D2 (ICLR 2019, r1lyTjAqYX).** *Contrast class for correction mechanisms.* Cite to show *why* Dreamer needs none (V-trace / stored-state+burn-in address real-data policy-lag that Dreamer's losses don't have).
11. **DreamerV2 — arXiv:2010.02193, `danijar/dreamerv2`.** *Continuity.* `train_every` per-step callback (default 5; Atari 16) confirms interleaving across the whole Dreamer line.

---

## Sources

- Hafner et al., *Dream to Control* (DreamerV1), arXiv:1912.01603 — Algorithm 1; "Environment interaction" appendix. Repo `danijar/dreamer@master` `dreamer.py`.
- Hafner et al., *Mastering Atari with Discrete World Models* (DreamerV2), arXiv:2010.02193. Repo `danijar/dreamerv2@main` `dreamerv2/train.py`, `dreamerv2/configs.yaml`.
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3), arXiv:2301.04104 / Nature 2025 — "Experience replay" / "replay ratio". Repos `danijar/dreamerv3@main` (`dreamerv3/configs.yaml`, `embodied/run/train.py`), `danijar/elements@main` (`elements/when.py:26-43`).
- Wu, Escontrela, Hafner, Goldberg, Abbeel, *DayDreamer*, arXiv:2206.14176 — §2–3.
- Fedus et al., *Revisiting Fundamentals of Experience Replay*, ICML 2020, arXiv:2007.06700.
- Chen, Wang, Zhou, Ross, *REDQ*, ICLR 2021, arXiv:2101.05982 — §3 (UTD, G=20).
- D'Oro, Schwarzer, Nikishin, Bacon, Bellemare, Courville, *Breaking the Replay Ratio Barrier*, ICLR 2023, OpenReview 0pC-9aBBVJe; ref impl `proceduralia/high_replay_ratio_continuous_control`.
- Nikishin, Schwarzer, D'Oro, Bacon, Courville, *The Primacy Bias in Deep RL*, ICML 2022, arXiv:2205.07802.
- Espeholt et al., *IMPALA*, ICML 2018, arXiv:1802.01561 — V-trace.
- Kapturowski, Ostrovski, Quan, Munos, Dabney, *R2D2*, ICLR 2019, OpenReview r1lyTjAqYX — representational drift, stored-state, burn-in.
- Wijmans et al., *DD-PPO*, ICLR 2020, arXiv:1911.00357.
- (Applications, not schedule-detailed) *DreamerV3 Framework for 3D Zero-Shot Goal Navigation*, dl.acm.org/10.1145/3800227.3800282; *WMNav*, arXiv:2503.02247.

## Open questions

- **No primary source directly ablates train-step scheduling (per-step interleave vs. episode-boundary defer vs. async) at a *fixed ratio* on Habitat ObjectNav** — the burstiness/staleness argument above is *derived* from the replay-ratio literature, not measured on ObjectNav. A small self-run ablation (same `train_ratio`, interleave vs. defer) would be the cleanest evidence for the thesis.
- **Exact per-benchmark replay ratios for DreamerV3 across *every* domain in the Nature version** were read from the repo `configs.yaml`, not a paper table; the paper states only that RR is "chosen to fit the step budget." Confirm any Habitat-specific ratio empirically.
- **The DreamerV1 released-code defaults** (`train_every`, `train_steps` in `danijar/dreamer`) were read via an automated fetch of `dreamer.py`; if a precise citation is needed, re-verify the exact default integers against the pinned file (they are consistent with the appendix's "100 updates / episode" but the raw ints should be quoted from source for the thesis).
- **D'Oro et al.'s exact maximum RR values** (per benchmark) could not be pulled from OpenReview (bot wall); the `updates_per_step 32` figure is from the reference-impl README. Verify against the camera-ready PDF if you cite specific RR ceilings.
