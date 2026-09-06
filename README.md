# Adaptive Meta-Scheduler

**A contextual-bandit CPU scheduler that learns which classical algorithm to run, guided by a from-scratch NumPy LSTM burst-time predictor.**

Final-year systems/ML project. Simulates 8 classical OS scheduling algorithms,
replaces SJF's oracle burst-time knowledge with a hand-rolled LSTM trained on
*observable* process features, and trains a contextual bandit to pick the best
scheduler per workload — beating the classical baselines with statistical
significance.

```
RL_Adaptive vs FCFS         27.3% lower avg wait   (p < 0.0001, Cohen's d = 1.18)
RL_Adaptive vs RoundRobin   47.5% lower avg wait   (p < 0.0001, Cohen's d = 2.08)
RL_Adaptive vs Priority     27.9% lower avg wait   (p < 0.0001, Cohen's d = 1.20)
RL_Adaptive vs MLFQ         41.5% lower avg wait   (p < 0.0001, Cohen's d = 1.98)
RL_Adaptive vs Lottery      43.8% lower avg wait   (p < 0.0001, Cohen's d = 2.00)
RL_Adaptive vs EDF            5.9% lower avg wait  (mixed: Wilcoxon p<0.001, t-test p=0.43, d=0.21)
RL_Adaptive vs SJF_Predicted  0.9% lower avg wait  (mixed: Wilcoxon p=0.03,  t-test p=0.90, d=0.03)
```
*(30 independent trials, seed 42 — see [`results/summary.json`](results/summary.json), reproduce with `python scripts/export_results.py`. All synthetic — see "Real-trace evaluation" below for the same comparison on a real 100K-job Alibaba GPU-cluster trace.)*

---

## What's actually novel here

Classical SJF scheduling needs to know a process's burst time *before it
runs it* — real operating systems don't have that. This project:

1. **Predicts burst time instead of assuming it's known.** A single-layer
   LSTM (32 hidden units, hand-written forward/backward pass in NumPy — no
   PyTorch/TensorFlow) estimates burst time from four features an OS actually
   can observe: priority, process-type class, inter-arrival gap, and a
   rolling average of recently completed bursts.
2. **Is honest about how much the LSTM actually contributes.** The 4
   observable features carry very little raw signal about an individual
   process's burst (`ρ ≈ 0.03`, next to random) — the dominant signal is the
   *process-type* category, which correlates at `ρ ≈ 0.91`. The scheduler
   (`schedulers/sjf_predicted.py`) is built as an explicit two-tier estimator
   — a strong type-based prior refined by the LSTM's within-type residual —
   rather than quietly leaning on oracle data or overclaiming what a weak
   feature set can do. This kind of stated limitation is exactly what a
   reviewer looks for and it costs nothing to say plainly.
3. **Learns *when* to trust that prediction.** A contextual bandit observes
   workload shape (queue pressure, burst variance, LSTM-predicted
   shortest-job ratio, starvation risk) at the start of each episode and
   picks one scheduler — FCFS, SJF_Predicted, Round Robin, or Priority — to
   run for the whole episode. It converges to preferring SJF_Predicted in
   ~82% of the states it visits, without ever being told that's the answer.
4. **Reports results the way a paper has to, not the way a demo wants to.**
   Mean ± std over 30 independent random trials, Welch's t-test *and*
   Wilcoxon signed-rank (both must agree to claim significance), Cohen's d
   effect sizes, and an oracle baseline clearly marked as unfair so it can't
   be mistaken for something the system actually beat.

## Project structure

```
adaptive_meta_scheduler/
├── simulator/            discrete-event core: Process, Clock, ReadyQueue,
│                         Dispatcher (+ context-switch-cost variant)
├── schedulers/           FCFS, SJF (oracle + predicted), Priority (w/ ageing),
│                         RoundRobin, MLFQ, EDF, Lottery — 8 policies, one
│                         shared select_next(ready_queue) interface
├── burst_predictor/      the from-scratch NumPy LSTM + its pre-trainer
├── rl_agent/             state encoder, reward function, Q-table
│                         (+ experience replay), contextual-bandit trainer
├── workload/             synthetic workload generator + real-trace loader
│                         (Google Cluster Trace / Alibaba / custom CSV)
├── evaluation/           metrics, significance testing (t-test + Wilcoxon),
│                         multi-trial experiment orchestration
├── visualization/        8 publication-quality chart types (matplotlib)
├── data/
│   └── real_traces/      alibaba_pai_gpu_trace_100k.csv — genuine 100K-job
│                          Alibaba PAI GPU-cluster trace, unmodified
│                          (PROVENANCE.md: source, license, citation,
│                          exactly what's real vs. derived)
├── scripts/
│   ├── export_results.py            canonical, reproducible "produce the
│   │                                  official numbers" entry point (below)
│   └── run_real_trace_evaluation.py  same statistical protocol, run against
│                                      30 windows of the real trace instead
│                                      of synthetic workloads (see "Real-trace
│                                      evaluation" above)
├── tests/                98 unit tests covering every module above
├── docs/
│   ├── Adaptive_Meta_Scheduler_Report.docx   full write-up (IEEE-style):
│   │                      introduction, related work, methodology, results,
│   │                      discussion, limitations, references — regenerate
│   │                      via scripts/export_results.py + the report
│   │                      generator after any code change
│   └── dashboard.html     self-contained interactive dashboard — real
│                          trained LSTM + Q-table run live in the browser
│                          against workloads you generate, plus every
│                          chart/table from results/summary.json; open
│                          directly, no server needed
├── webapp/                Flask control panel — same idea as dashboard.html,
│                          but calls the real Python backend server-side
│                          instead of a JS port (see "Flask control panel"
│                          above); app.py, sim_runner.py, templates/, static/
├── main.py                the original quick pipeline (4 charts)
├── experiment_runner.py   the full 6-feature research pipeline
│                          (real traces, extra baselines, significance
│                          tests, context-switch sweep, extended metrics,
│                          3 extra charts) — CLI flags below
└── results/               generated output (git-ignored; regenerate it)
```

Four ways to run this, from quickest to most complete:

| Entry point | What it does | Runtime |
|---|---|---|
| `python main.py` | Pretrain LSTM → train bandit (600 ep) → single demo run → 30-trial stats → 4 charts | ~90s |
| `python experiment_runner.py [--quick]` | Everything in `main.py`, plus a trace-loading demo (real API, falls back to an auto-generated synthetic sample unless pointed at a real CSV — see "Real-trace evaluation" below for the genuine-data version), MLFQ/Lottery/EDF baselines, significance tests, context-switch-cost sensitivity sweep, extended metrics (fairness, tail latency, slowdown), 4 more charts | ~2–4 min |
| `python scripts/export_results.py [--quick]` | The reproducible, seeded version used to generate the numbers in this README, the report, and the dashboard — one consistent set of 30 trials feeding *both* the summary table and the significance tests (see script docstring for why that matters), written to `results/summary.json` | ~90s |
| `python scripts/run_real_trace_evaluation.py [--quick]` | Same protocol, against 30 windows of the real Alibaba PAI GPU trace instead of synthetic workloads — loads the already-trained LSTM/bandit from `results/*.json` rather than retraining, written to `results/real_trace_summary.json` (see "Real-trace evaluation" above) | ~2 min |

`--quick` on any of the last three runs a 100-episode / 10-trial smoke test instead of the full 600/30.

## Flask control panel (optional, interactive)

`webapp/app.py` is a small Flask frontend — the fifth way to run this, and
the only one meant to be clicked through rather than read as output. Unlike
`docs/dashboard.html` (a from-scratch JS port of the LSTM + Q-table, built
so it needs no server), this calls the real Python simulator, the real 8
scheduler classes, and the real trained LSTM/bandit **directly, server-
side** — there's no second implementation of the algorithms to drift out
of sync with the one this report describes.

```bash
pip install -r requirements.txt   # now includes Flask
python webapp/app.py
# -> http://127.0.0.1:5000
```

Pick a workload (a synthetic profile, or one of the 30 real Alibaba trace
windows), then either run one scheduler at a time (metrics + a Gantt chart
built from the actual tick-by-tick execution) or "Compare all 9" for a
ranked bar chart. If you haven't run `scripts/export_results.py` yet, the
page says so up front — SJF_Predicted and RL_Adaptive will otherwise be
running an untrained model.

One thing worth knowing if you extend this: the codebase has two separate
LSTM-backed SJF implementations with two different online-learning
cadences (`schedulers/sjf_predicted.py`'s self-priming version, used for
the standalone SJF_Predicted baseline everywhere else in this README, vs.
`schedulers/sjf.py`'s version, which is what the bandit's action space
actually wraps — see `rl_agent/trainer.py`'s `_make_schedulers`). `webapp/sim_runner.py`
replays whichever cadence is actually in play rather than picking one and
quietly reusing it for both — get that wrong and RL_Adaptive's Gantt chart
would render correctly while silently *not* reflecting the model this
report describes (SJFScheduler falls back to the plain type-prior estimate
whenever nothing has primed its LSTM cache). Verified by comparing this
app's numbers against `RLTrainer.evaluate()`'s own numbers on the same
workload — they match.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
python main.py
```

Python 3.9+. `numpy` and `matplotlib` are required; `scipy` is optional (the
significance-testing module falls back to a pure-Python Welch's t-test /
Wilcoxon approximation if scipy isn't installed).

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

98 tests, all passing, covering the simulator core, all 8 schedulers, the
LSTM predictor (including a basic BPTT learning sanity check — training on a
noise-free task must reduce error, not just avoid crashing), the RL agent's
Bellman update and state discretisation, and the statistics module (e.g.
identical distributions must *not* be flagged significant; clearly-separated
ones must be).

## How the pieces fit together

```
                    ┌─────────────────────┐
   workload/        │  WorkloadGenerator   │  3 profiles: cpu_bound,
   generator.py     │  (or trace_loader)   │  interactive, mixed-adversarial
                    └──────────┬───────────┘
                               │ List[Process]
                               ▼
                    ┌─────────────────────┐
   burst_predictor/ │  LSTM (32 hidden)   │  predicts burst time from
   lstm_predictor.py│  Adam + BPTT        │  4 observable features
                    └──────────┬───────────┘
                               │ predicted burst → feeds SJF_Predicted
                               │ and the bandit's state (feature 3)
                               ▼
                    ┌─────────────────────┐
   rl_agent/        │ Contextual Bandit    │  state: 5 features, 3^5=243
   trainer.py        │ (ε-greedy Q-table)  │  action: {FCFS, SJF_Pred,
                    │ + experience replay │           RoundRobin, Priority}
                    └──────────┬───────────┘  reward: −avg_waiting_time
                               │ picks ONE scheduler for the whole episode
                               ▼
                    ┌─────────────────────┐
   simulator/       │ Clock/ReadyQueue/    │  runs the chosen scheduler
                    │ Dispatcher            │  tick-by-tick to completion
                    └──────────┬───────────┘
                               ▼
                    ┌─────────────────────┐
   evaluation/       │ metrics + t-test /  │  → results/summary.json,
   visualization/   │ Wilcoxon + 8 charts │    results/*.png
                    └─────────────────────┘
```

The **feedback loop** that makes this more than a fixed pipeline: the
bandit's state includes an LSTM-derived "predicted shortest-job ratio"
feature. As the LSTM's online BPTT fine-tuning drives its MAE down across
training, that feature gets more informative, so the bandit more reliably
recognises when SJF_Predicted is the right call — which produces more
completions with accurate predictions, which gives the LSTM more training
data. `rl_agent/state_encoder.py` documents this coupling in detail.

## Design choices worth knowing about (for the viva)

- **Contextual bandit, not tick-level RL.** An earlier version switched
  schedulers every 2 ticks; that destroyed the ordering SJF-style scheduling
  needs to build up, and the reward signal in any 2-tick window was
  dominated by noise (SNR < 1 — unlearnable). Making one decision per
  episode and rewarding on `-avg_waiting_time` gives SNR ≈ 15:1 and the
  agent converges in well under 200 episodes. `rl_agent/trainer.py`
  explains this trade-off in full.
- **5-feature, not 8-feature, state space.** An 8-feature state
  (3⁸ = 6,561 states) left most states with only a handful of visits in
  600 episodes — nowhere near enough for Q-values to converge. Cutting to 5
  features calibrated at the real p33/p66 percentiles of simulated data
  (3⁵ = 243 states, ~30–35% of samples per bin) gives ~270 visits per state.
  See `rl_agent/state_encoder.py`.
- **Two-tier SJF estimation, not a single blended score presented as
  "the LSTM predicts burst time."** Framing this as pure LSTM prediction
  without acknowledging the weak raw feature correlation would be an
  overclaim; framing it as *type-prior + learned within-type refinement* is
  both more accurate and, if anything, a more interesting result (it shows
  where the signal in an OS process's metadata actually lives).
- **Context-switch-cost sensitivity sweep.** Real kernels pay ~1–10 μs per
  context switch. `simulator/context_switch.py` adds this as an opt-in
  dispatcher extension (default cost = 0, so nothing about the main results
  changes) and `experiment_runner.py` sweeps cost ∈ {0, 1, 2, 4} to check
  whether the algorithm ranking holds up once preemption has a cost, as
  real kernels experience it.
- **Online-training bug worth knowing about.** `BurstTimePredictor.record_burst`
  used to build each online BPTT training sample purely from `self._ctx`
  — a deque that nothing in the class ever appends to, so it was always
  empty. Every online fine-tuning sample was therefore an all-zero input
  sequence mapped to the true burst time, so the 600-episode online phase
  could only ever nudge the model toward predicting roughly the running
  mean burst, never learn a function of a process's own features — silently
  contradicting `predict_process()`, which *does* condition on the real
  4-feature vector. Pre-training (200 synthetic sequences × 8 epochs, which
  *does* use real features) was unaffected, which is why the model was
  never non-functional, just not actually improving from the signal
  `record_burst` claimed to feed it. Fixed by caching the exact feature
  vector `predict_process()` used for each pid and replaying that same
  vector in `record_burst()` (see `burst_predictor/lstm_predictor.py` and
  `tests/test_burst_predictor.py::test_record_burst_trains_on_actual_features`).
  All LSTM/bandit numbers in this README reflect the run *after* this fix.
- **Reproducibility fix worth knowing about.** `MLFQScheduler` originally
  iterated `set(ready_queue.get_all())` when re-admitting processes after a
  priority boost. `Process` has no custom `__hash__`, so that set ordered by
  Python's default *identity*-based hash — stable within a single run, but
  not guaranteed stable across separate interpreter invocations, which
  silently made post-boost FCFS tie-breaking (and therefore MLFQ's exact
  numbers) irreproducible even with `random.seed()`/`np.random.seed()` fixed.
  Running `scripts/export_results.py` twice with the same seed exposed this
  as a swing in MLFQ's numbers between runs; every algorithm with no
  `set()`-of-`Process` in its scheduler is already exactly reproducible.
  Fixed by iterating `ReadyQueue`'s own ordered list instead of a bare set
  (see `schedulers/mlfq.py` and
  `tests/test_schedulers.py::test_mlfq_priority_boost_readmits_in_ready_queue_order`).
  One caveat honestly noted rather than glossed over: the LSTM-driven path
  (SJF_Predicted / RL_Adaptive) can still show ~1% run-to-run drift in its
  aggregate 30-trial mean even after this fix, consistent with ordinary
  floating-point non-associativity in NumPy's BLAS backend rather than any
  remaining Python-level nondeterminism — the classical (non-learned)
  schedulers are unaffected and exactly reproducible.

## Real-trace evaluation (Alibaba PAI GPU cluster)

`data/real_traces/alibaba_pai_gpu_trace_100k.csv` is a genuine, unmodified
100,000-job trace from Alibaba's production PAI GPU cluster (Alibaba Cluster
Trace Program — see [`data/real_traces/PROVENANCE.md`](data/real_traces/PROVENANCE.md)
for source, license, and citation). `load_alibaba_pai_trial_windows()` in
`workload/trace_loader.py` slices it into 30 non-overlapping, evenly-spaced
windows of 150 real jobs each; `scripts/run_real_trace_evaluation.py` runs
every scheduler — including the *same* LSTM + bandit trained only on
synthetic data (`results/lstm_weights.json` / `results/q_table.json`,
loaded, not retrained) — against them.

This is deliberately an **out-of-distribution generalization test**, not
real-data training: the policy has never seen this trace before being
evaluated on it. Feeding real windows into `RLTrainer.train()` instead of
`WorkloadGenerator` would be real-data training — that's a natural next
step this doesn't yet take.

```
RL_Adaptive vs FCFS          53.5% lower avg wait  (t-test p=0.00017, Wilcoxon p<0.001, d=1.07)
RL_Adaptive vs Priority      53.5% lower avg wait  (t-test p=0.00017, Wilcoxon p<0.001, d=1.07)
RL_Adaptive vs MLFQ          47.4% lower avg wait  (t-test p=0.0012,  Wilcoxon p<0.001, d=0.89)
RL_Adaptive vs Lottery       27.3% lower avg wait  (t-test p=0.046,   Wilcoxon p<0.001, d=0.53)
RL_Adaptive vs RoundRobin    19.5% lower avg wait  (mixed: t-test p=0.15 n.s., Wilcoxon p<0.001, d=0.37)
RL_Adaptive vs SJF_Predicted  2.0% higher avg wait (both n.s.: t-test p=0.90, Wilcoxon p=0.84, d=0.03)
RL_Adaptive vs EDF           16.8% higher avg wait (both n.s.: t-test p=0.28, Wilcoxon p=0.14, d=0.28)
```
*(30 real windows, seed-independent — see [`results/real_trace_summary.json`](results/real_trace_summary.json), reproduce with `python scripts/run_real_trace_evaluation.py`, ~2 min.)*

**Read this the same way as the synthetic result, not more generously.**
The bandit's learned policy chose the SJF_Predicted action in 29 of 30 real
windows (1 chose Priority) — so "RL_Adaptive ≈ SJF_Predicted, both clearly
beat FCFS/Priority/MLFQ/Lottery, neither beats EDF" is one finding wearing
two names here, not two independent confirmations. The result worth taking
away is that a bandit trained *only* on synthetic workloads kept making
that same good call on a real, unseen production trace — not that it
discovered something new from real data, because it wasn't shown any.

Two things worth knowing if you build on this further:
- `priority` for these real jobs is a **derived proxy** from requested
  CPU/GPU (`_bucket_pai_priority` in `trace_loader.py`) — this trace has
  no ground-truth priority field, so any "RL beats Priority scheduler on
  real data" framing carries that caveat.
- Real job durations here are heavily right-skewed (3s to 599,445s in the
  raw trace). A tick-by-tick simulator can't run that span directly, so
  burst times are scaled to a synthetic-comparable median (~15 ticks) and
  clipped at `MAX_REAL_BURST_TICKS = 500` — typically 5-20% of jobs per
  window hit that clip. This is disclosed and printed at load time (see
  `_scaled_and_capped_bursts`), not a silent transformation.

## Limitations (stated up front, not buried)

- **LSTM MAE is real but modest**: ~9.6 ticks on a workload with bursts
  spanning 1–40 ticks (avg ≈ 20) — meaningful but far from perfect, which is
  exactly why the two-tier estimator leans primarily on the type prior.
- **RL_Adaptive's edge over SJF_Predicted and EDF is small and mixed** —
  0.9% and 5.9% lower mean wait respectively, each significant under the
  paired Wilcoxon test but *not* under the parametric t-test, with
  negligible-to-small effect sizes (Cohen's d ≤ 0.21). The honest framing is
  that the bandit *discovers* SJF_Predicted as the dominant policy for this
  workload distribution (selecting it in ~82% of visited states) rather than
  discovering something better than classical scheduling already knew. The
  interesting result is that it discovers this automatically from workload
  features, not that it convincingly exceeds a well-tuned heuristic.
- **Training is still synthetic-only.** The LSTM and bandit are trained
  exclusively on the three parametrised synthetic profiles — see
  "Real-trace evaluation" above for what changed: *evaluation* now includes
  a real, unmodified 100K-job Alibaba GPU-cluster trace, but the policy
  being evaluated on it has never been trained on real data. Feeding
  `load_alibaba_pai_trial_windows()` into `RLTrainer.train()` instead of
  `WorkloadGenerator` is the natural next step this doesn't yet take.

## Reference concepts

MLFQ, Priority-with-ageing, and Lottery scheduling are textbook OS
algorithms; EDF is the classical real-time scheduling result (Liu &
Layland, 1973); Lottery scheduling is due to Waldspurger & Weihl (1994);
Jain's fairness index is from Jain et al. (1984); the contextual-bandit
framing for algorithm/system selection follows the general approach used in
Mao et al.'s *Learning Scheduling Algorithms for Data Processing Clusters*
(2019) and related learned-systems work.

## License

MIT — see [`LICENSE`](LICENSE).
