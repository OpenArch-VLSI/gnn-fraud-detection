# Camouflage-Robust Graph Neural Networks for Fraud Ring Identification — Project Roadmap

## 0. How to use this document

**You (the student):** Read it once top to bottom so you know the shape of the whole
project. After that, use it as a checklist — each phase has a "Definition of Done"
you can tick off. The "Concepts to understand" boxes are your personal study list;
you should be able to explain those in your own words before moving on, regardless
of what the agent writes for you.

**Antigravity (the coding agent), if you're reading this file:** Work through the
phases in order. Do not start Phase *N+1* until Phase *N*'s Definition of Done is
satisfied. Each phase's task list is your task list for that session. Obey every
"Agent guardrails" note literally — they exist because of constraints the professor
set, not stylistic preferences. If a request from the user conflicts with a
guardrail (e.g. "just import GATConv to save time"), flag the conflict instead of
silently complying or silently refusing. One more constraint that isn't the
professor's but is just as real: you don't have your own GPU. Every phase below
that needs one runs on Kaggle, not in your local sandbox — see §0.1 for exactly
where that line falls.

> **Non-negotiable rule for Phase 7:** do not port or closely translate code from
> the CARE-GNN or PC-GNN reference repositories. Read them, understand the
> mechanism, then design your own variant. This is a graded, original-work
> deliverable — copying a reference implementation defeats the point and risks
> an academic integrity problem, not just a weak grade.

## 0.1 Compute & tooling setup

No local high-spec machine is available for this project. Every GPU-bound phase
runs on Kaggle's free notebook tier, and the coding agent is Gemini 3.1 Pro (High
thinking level) running inside Google's Antigravity IDE.

- **Local vs. remote split (applies everywhere):** Antigravity/Gemini runs on
  your machine (or wherever the IDE is hosted) and has no GPU of its own. It's
  full-time help for writing code, designing the Phase 7 mechanism, debugging,
  and drafting the report — but the actual execution of every GPU-bound phase
  (3, 4, 5, 7, 8) happens on Kaggle, a separate environment it can only reach by
  pushing code in (via `git clone` inside the notebook, or the `kaggle` CLI) and
  pulling results back out, not by running things directly the way it would
  against a local GPU.
- **Debug locally first, always.** Because Antigravity cannot watch a Kaggle
  run live or fix a crash mid-session, the default workflow for every
  GPU-bound phase is: write and iterate against a small local CPU subsample
  until the script is confirmed correct (runs cleanly, loss behaves, no shape
  errors), *then* hand it off to Kaggle for the real run. Kaggle time is a
  rationed resource — treat it as reserved for confirmed-stable code, not a
  place to debug. If several small variants need testing (e.g. a few Phase 7
  mechanism tweaks), queue them within one Kaggle session rather than starting
  a fresh session per idea, since the quota is billed per hour, not per run.
- **Thinking-level note:** run Gemini 3.1 Pro on "High" throughout — quota
  resets every 5 hours, so there's no real cost to keeping the deeper
  reasoning on for the whole project rather than switching levels per task.
- **Phase 0** is built around Kaggle account/accelerator/quota setup and a
  git-to-Kaggle sync method.
- **Phase 1 / Phase 2** run on a CPU-only session (Kaggle or local) — no GPU
  needed for EDA or graph construction, and CPU sessions don't draw down your
  GPU quota.
- **Phase 3:** assume Kaggle's free GPUs are meaningfully small (16GB VRAM,
  ~29GB system RAM once a GPU is attached). Neighbor sampling is the likely
  default, not a fallback you probably won't need.
- **Phase 5, 7, 8:** include checkpoint/resume and GPU-hour budgeting
  guardrails — a lost multi-hour run against a 30-hour weekly cap is expensive
  to redo.
- **Appendix E** includes risk-register rows for quota exhaustion, session
  disconnects, and local/Kaggle environment drift.

---

## 1. Project summary

Build a fraud detector over a transaction graph (accounts/transactions as nodes,
relationships as edges) using Graph Neural Networks. Part 1 hand-codes GraphSAGE
and GAT from scratch as a foundation. Part 2 adds one genuinely new piece: a
mechanism that resists "camouflage" — fraud rings that deliberately connect
themselves to normal accounts to look legitimate. This is the actively-studied gap
in this field right now (see Appendix D).

**Novelty, precisely stated:** camouflage-resistant GNNs were originally built
and tested on review-fraud graphs (Yelp/Amazon), but transaction-graph
benchmarks for this exact problem already exist too (T-Finance, T-Social,
S-FFSD — Appendix D), so "first to apply this to transaction data" is not an
accurate or defensible claim — don't pitch it that way to your professor or in
the report. What's still genuinely open on **IEEE-CIS specifically**: no prior
work runs a CARE-GNN/PC-GNN-style explicit neighbor-filtering mechanism against
from-scratch GraphSAGE/GAT baselines with a proper multi-seed ablation on this
dataset, isolating how much of any gain comes from camouflage-resistance
specifically versus attention alone. The closest existing work (Appendix D —
RL-GNN, 2025) combines GAT with an RL controller directly on IEEE-CIS and
reports 0.872 AUROC — a useful external number to benchmark against in Phase 8,
and a paper you need to explicitly differentiate from in your report (it
doesn't target camouflage/heterophily resistance specifically, and doesn't
ablate against baselines you built yourself).

## 1.1 Current status (as of latest update)

Phases 0–4 and 6 are complete. Phase 5 (baselines) and Phase 7 (novel module)
are both **in progress, not done**, despite real training runs having
happened on Kaggle:

- **SAGE**: trained successfully, twice (original graph and NaN-imputation-fixed
  graph). Best result: PR-AUC 0.4147, ROC-AUC 0.8618 (rebuilt graph, epoch 15).
- **GAT**: trained successfully once, after fixing an instability bug (grad
  clipping + lower LR). Best result: PR-AUC 0.3157, ROC-AUC 0.8103 (original
  graph, epoch 6) — still behind SAGE. **Not yet rerun against the corrected
  graph** — this is an open task, not finished.
- **Camouflage-GAT**: attempted once on the full graph, **collapsed to
  random-chance performance** (ROC-AUC ≈ 0.5) and was cancelled. Root cause
  not yet diagnosed — leading suspects are the untuned `aux_loss_weight` or a
  bug in the trust-signal computation, neither confirmed.
- **Tabular reference baseline**: not started.
- **Test-set evaluation**: `evaluate.py` had a checkpoint-unwrapping bug that
  made it completely non-functional; the bug is fixed, but evaluation has
  never actually been run — every number above is a validation-set number
  from training, not a test-set number.
- Along the way: a `load_config()` bug (silently discarding YAML in favor of
  CLI defaults) and a NaN-imputation bug in `build_graph.py` were also found
  and fixed, the latter requiring a full graph rebuild.

## 2. Top-level definition of done

- [ ] From-scratch GraphSAGE and GAT baselines, trained and evaluated on a real
      fraud dataset, with imbalance-aware metrics (not accuracy) — **partially
      done**: both trained with real val PR-AUC/ROC-AUC results logged, but
      GAT needs a rerun on the corrected graph, and neither has been run
      through test-set `evaluate.py` yet
- [ ] A non-graph reference baseline, so "the graph helped" is a measured
      result and not an assumption — not started
- [ ] At least one experiment testing robustness to *escalating* camouflage,
      not just performance at the dataset's fixed, natural camouflage level
      — not started (Phase 8)
- [ ] One clearly-scoped novel extension, implemented, ablated, and compared
      fairly against the baselines on identical splits/seeds — module is
      implemented, but its only full-graph training run collapsed to
      random-chance performance and hasn't been fixed or compared yet
- [ ] A codebase a stranger could clone and reproduce your headline number from
- [ ] A written report: motivation, related work, method, results, honest
      limitations — related-work piece done (Phase 6); results/method/report
      writing not started
- [ ] All of the above fits inside your actual semester timeline

## 3. Tech stack

- Python 3.10+, PyTorch. On Kaggle you don't pick a CUDA build yourself — the
  notebook image ships a fixed PyTorch+CUDA pair. Check `torch.__version__`
  first thing in Phase 0 and treat it as given; for local editing/unit-testing
  with Antigravity (no GPU available there), install CPU-only PyTorch in a
  small venv, same major version where possible, purely so tests and
  small-sample debugging run instantly without touching Kaggle quota
- Scatter/reduce ops: prefer **native PyTorch** (`torch.Tensor.scatter_reduce_`,
  `index_add_`) over the separate `torch_scatter` package. `torch_scatter`'s own
  maintainers note most of its functionality now lives in PyTorch directly, and
  the package is a common source of install pain (exact CUDA/torch/OS wheel
  matching). At this graph scale (≤~600K nodes) writing scatter-mean and
  scatter-softmax yourself with native ops is both more reliable and more in
  the spirit of Appendix C's "from scratch" scope — fall back to `torch_scatter`
  only if you hit a specific performance wall
- PyTorch Geometric **only** for its `Data`/`Dataset` container and any dataset
  download helpers (e.g. `EllipticBitcoinDataset`) — not for its `nn` layers.
  Install it with `!pip install torch_geometric` in your first Kaggle cell
  each session and confirm it imports cleanly against the pre-installed
  `torch` version before writing anything that depends on it — this is the
  same CUDA/torch/PyG wheel-matching risk §3 already flags, just against
  Kaggle's fixed image instead of your own driver
- pandas / numpy / scikit-learn for tabular EDA and metrics
- matplotlib for figures
- `lightgbm` or `xgboost` (or plain `sklearn.linear_model` if you'd rather
  avoid another dependency) for the Phase 5 tabular reference baseline

## 4. Repository structure

```
fraud-gnn/
  .github/
    workflows/              # CI: run tests/ on every push (Phase 10)
  data/
    raw/                  # untouched downloads
    processed/             # serialized graph objects
  src/
    data/                  # graph construction scripts
    models/
      layers/              # your from-scratch SAGEConv, GATConv
      baselines.py          # GraphSAGE + GAT; also the non-graph tabular baseline
      camo_module.py       # your novel Part 2 piece
    train.py
    evaluate.py
    utils/
  notebooks/                # EDA, plus a thin Kaggle entry-point notebook (clones this repo, calls src/train.py) — never production logic itself
  experiments/               # one config + result log per run
  tests/                     # unit tests for your hand-written layers
  report/
    figures/
  README.md
  requirements.txt
  ROADMAP.md                 # this file
```

This repo lives on GitHub and is what Antigravity edits directly. Kaggle never
edits it — a notebook only pulls it in (`!git clone`, or a token-authenticated
pull for a private repo) at the start of a session, runs `src/train.py`, and
its outputs get pulled back out (commit the notebook version, or push results
to a Kaggle Dataset to persist beyond one session). Keep it one-directional:
code changes always originate locally with Antigravity, never inside the
Kaggle notebook itself, or the two copies will drift.

---

## Phase 0 — Environment & repo setup
**~2–3 days**

Goal: a reproducible environment and a scaffold, before any modeling. This spans
two places — a local, GPU-less environment where Antigravity actually edits
code, and Kaggle, where the code actually runs. Set both up and prove they talk
to each other before Phase 1.

Tasks

*Local side (Antigravity's workspace):*
- [ ] Initialize git; create the folder structure in §4
- [ ] Create a lightweight local venv with **CPU-only** PyTorch — enough to run
      unit tests and debug small-sample code with Antigravity without needing a
      GPU or touching Kaggle quota
- [ ] Write a `set_seed(seed)` utility used everywhere (Python, numpy, torch, cuda)
- [ ] Stub `README.md` with the one-paragraph project summary

*Kaggle side (where every GPU-bound phase actually executes):*
- [ ] Create a Kaggle account and complete phone verification — required
      before GPU/TPU accelerators or internet access are unlocked in notebooks
- [ ] Open a new Notebook; set Accelerator → GPU (P100's single 16GB device is
      the simplest default; T4 x2 gives two separate 16GB GPUs, only useful if
      you deliberately code for two devices) and Internet → On
- [ ] In a cell: `import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_properties(0).total_memory)`
      — confirm the GPU is visible; expect ~16GB, not 24GB
- [ ] Decide and test your git-to-Kaggle sync method now, on a trivial script,
      before Phase 1 needs it for real: `!git clone` your GitHub repo into the
      notebook (use **Kaggle Secrets** for the token if it's private — do not
      embed the token directly in the notebook cell, as notebook version history
      will permanently leak it if the notebook is ever made public), or push/pull
      via the `kaggle` CLI's `kernels push` / `kernels pull`. Confirm a full cycle
      works — edit locally with Antigravity → sync to Kaggle → run → pull
      results back — before you build anything on top of it
- [ ] Check your current weekly GPU-hour and TPU-hour balance in Kaggle's
      Settings, so Phase 1 onward starts from a known budget
- [ ] Record the exact pre-installed `torch`/CUDA versions from the check above
      in `requirements.txt`, so `pip install torch-geometric` later targets a
      version that actually matches Kaggle's image, not a guess

Concepts to understand: why seeding every RNG source (not just
`torch.manual_seed`) matters for reproducible results; why a
`torch-geometric`/`torch_scatter` wheel has to match the *notebook's*
pre-installed PyTorch+CUDA build rather than your own — you don't choose the
CUDA driver, Kaggle does, so the mismatch risk is "your pip install vs.
Kaggle's fixed image."

Agent guardrails: pin exact package versions in `requirements.txt` as you
install them — don't let this drift silently later. Before kicking off any
GPU session, say what's about to run and your best guess at how long, and get
a go-ahead first — a burned session against a 30-hour weekly cap is more
expensive to redo than a wrong line of code.

Definition of Done: local unit-test environment runs (CPU only); a Kaggle
notebook with GPU attached prints `True` and a ~16GB figure; your sync method
reproduces a trivial edit-sync-run-pull cycle end-to-end; scaffold folders
committed; current GPU/TPU quota balance checked.

---

## Phase 1 — Problem framing & dataset decision
**~3–4 days**

Goal: lock the dataset and write down precisely what you're building, before
writing model code.

**Dataset: IEEE-CIS** (full comparison against the Elliptic Bitcoin
alternative that was considered and passed over is in Appendix A). IEEE-CIS
was chosen because constructing the multi-relation graph yourself is real
learning *and* sets up the camouflage angle in Phase 7 naturally — a fraud
ring sharing a device or card is a direct camouflage signal.

Tasks
- [x] Attach the dataset as a notebook input: the official **"IEEE-CIS Fraud
      Detection"** Kaggle competition (join the competition's rules first if
      prompted, it's free) rather than downloading it anywhere yourself or
      using a third-party reupload
- [ ] Do this phase's EDA on a **CPU-only** Kaggle session (Accelerator → None)
      — it doesn't need a GPU, and CPU sessions don't draw down your 30-hour
      weekly GPU quota
- [ ] EDA notebook: class balance, missing values, feature types; check the
      cardinality of candidate "shared entity" columns (`card1`–`card6`,
      `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo`) — this
      is what the Phase 2 relation-column decision below is based on
- [ ] Write a one-paragraph problem statement and a one-paragraph novelty
      statement — this becomes tomorrow's pitch to your professor and later the
      intro of your report

Concepts to understand: why accuracy is a bad headline metric under class
imbalance — concretely, a model that always predicts "not fraud" already
scores ~96.5% on IEEE-CIS (20,663 fraud / 590,540 total); why a **random**
train/test split can leak information in transaction data (better: a
time-based split).

Agent guardrails: this phase is analysis only — no training yet.

Definition of Done: EDA notebook committed; problem + novelty statements written;
dataset choice locked in `README.md`.

---

## Phase 2 — Graph construction pipeline
**~1 week**

Goal: a deterministic script that turns raw data into a graph object your models
can consume, run once, reused by every later phase.

- [x] Node scope decided: **transactions as the only node type** (the
      simpler option) — shared entities (card, device, email) become
      *relation types*, i.e. edges, rather than their own separate node
      types in a fully heterogeneous graph.
- [x] Build one edge relation per shared-entity type. **Final relation
      columns: `card1`, `card2`, `addr1`, `P_emaildomain`, `DeviceInfo`.**
      This mirrors CARE-GNN's multi-relation design (it used relations like
      same-user / same-time / same-star on review data) — the same idea
      applied to transaction data. `addr2` was evaluated and **deliberately
      excluded** from both edges and node features: it has one dominant
      value and an extremely large average group size (~175K), making it
      low-signal and prone to the same hub-node problem described below if
      used as an edge relation.
- [x] **Cap degree on hub entities.** A shared value like a common email
      domain would otherwise connect a huge fraction of all transactions —
      this silently turns the graph into a near-clique. **`DEGREE_CAP = 20`**
      (an explicit, commented config constant in `build_graph.py`): groups of
      size ≤ 20 are fully connected; larger groups have each node randomly
      sample up to 20 other group members rather than being fully connected
      or dropped entirely. (`DEGREE_CAP` was originally tried at 100, which
      produced an impractical ~273M edges / ~9GB graph file; 20 cuts
      capped-group edge volume by roughly 5x while still preserving the
      relation signal.)
- [x] Serialize: combined `edge_index` tensor (all five relations merged),
      node feature matrix, label vector → `data/processed/graph.pt`, so
      nothing downstream re-parses raw CSVs. **Verified output: 590,540
      nodes, 425 features, 45,528,526 total combined edges**
      (card1: 11,200,772; card2: 11,631,988; addr1: 10,486,078;
      P_emaildomain: 9,921,680; DeviceInfo: 2,288,008).
- [ ] Write one explicit sentence in `README.md` stating whether the setup is
      **transductive** (the full graph is visible at train time, and only
      labels are split) or **inductive** (test-period nodes are entirely
      absent from the graph during training) — this hasn't been written down
      explicitly yet and should be, so a reader doesn't have to
      reverse-engineer it from the code.

Concepts to understand: `edge_index` vs. dense adjacency representation; why hub
nodes distort message passing; transductive vs. inductive setting.

Agent guardrails: every threshold or cap you pick (degree cap, which columns count
as "shared entity") goes into a config value with a comment explaining the choice
— not a magic number buried in code. This script must be re-runnable end-to-end
from one command. Run this phase on a CPU-only session too — graph construction
is data engineering, not model training, and doesn't need GPU quota.

Definition of Done: `python src/data/build_graph.py` goes from raw file to saved
graph object in one run, printing node count, edge count per relation, and class
balance at the end.

---

## Phase 3 — GraphSAGE from scratch
**~1 week**

Goal: hand-implement mean-aggregation message passing and a 2-layer SAGE model.

Concepts to understand before coding:
- Neighbor sampling, and why full-batch training doesn't scale to large graphs
  in general. Kaggle's free GPUs are a small box (16GB VRAM, ~29GB system RAM
  once a GPU is attached), and PC-GNN's 128GB-RAM comparison point is ~4.5x
  more RAM than you actually have. With ~590K nodes and ~45.5M edges,
  full-batch training on the whole IEEE-CIS graph is not viable on this
  hardware — neighbor sampling (via `NeighborLoader`, see Phase 5) is the
  actual approach used, not just a likely default
- The update rule: new embedding for node *v* = `σ(W · CONCAT(h_v, AGG({h_u for u in neighbors(v)})))`
- Why the "mean aggregator" is *not* the same as the GCN aggregator (different
  self-loop and normalization handling)

Tasks
- [x] Implement a `SAGEConv` layer by hand using primitive tensor ops (see
      Appendix C for what "by hand" allows)
- [ ] Decide explicitly what your mean aggregator does with a zero-degree node
      (mean of an empty neighbor set is undefined) — a self-loop or a small
      learned "isolated-node" fallback vector are the two standard fixes.
      Double-check your Phase 2 degree caps don't quietly create zero-degree
      nodes you haven't accounted for.
- [x] Stack two layers (`sage.py`, with BatchNorm + ReLU + dropout), add a
      binary classification head (single-logit output)
- [x] Unit tests exist and pass (9/9 in the current suite), covering the
      hand-written layers' shapes and gradient flow

Agent guardrails: **do not** import `torch_geometric.nn.SAGEConv`,
`dgl.nn.SAGEConv`, or any prebuilt message-passing layer. Comment each line of
the layer with which part of the formula above it implements. **Keep this
layer as the single canonical `SAGEConv` in the repo** — at one point two
independently-written versions (from different contributors) existed side by
side, which risks shape mismatches and checkpoints that don't correspond to
the code loading them; resolve any future duplication immediately rather than
letting two versions drift.

Definition of Done: unit tests pass; trains without NaN loss on a small
subsample within a few minutes.

---

## Phase 4 — GAT from scratch
**~1 week**

Goal: hand-implement attention-based aggregation.

Concepts to understand: the attention-coefficient formula
`e_ij = LeakyReLU(a^T [W·h_i || W·h_j])`, softmax-normalized over each node's
neighborhood; multi-head attention as several independent attention computations
concatenated together.

Tasks
- [x] Implement `GATConv` by hand, with multi-head attention and explicit
      self-loops (so a node's own features always contribute to its own
      update)
- [x] In the hand-written per-neighborhood softmax, subtract the max logit
      before exponentiating (the standard numerically-stable softmax trick) —
      implemented, guarding against silent NaNs on the degree-capped hub
      nodes from Phase 2
- [x] Unit tests exist and pass, including attention-weights-sum-to-1 checks

Agent guardrails: same rule as Phase 3 — no `GATConv` import, ever. Same
canonical-single-implementation note as Phase 3 applies here too.

Definition of Done: attention-sums-to-1 test passes; training is stable (loss
decreases, no NaNs).

---

## Phase 5 — Baseline training & evaluation harness
**~4–5.5 days**

Goal: a rigorous, reusable train/eval loop *before* touching the novel idea, so
Phase 7 has a trustworthy number to beat.

Tasks
- [x] Handle class imbalance: `pos_weight` inside `BCEWithLogitsLoss`,
      computed only from the training split (no leakage from val/test)
- [x] Metrics: ROC-AUC, PR-AUC, and F1 are tracked (accuracy is
      intentionally not used as the headline number, since a
      not-fraud-always model already scores ~96.5% on IEEE-CIS)
- [x] Config-driven `train.py --model <sage|gat|camouflage> --epochs N`,
      with `NeighborLoader` mini-batching (fanout `[15, 15]`) so training
      fits Kaggle's ~16GB GPU against the graph's 45.5M edges; best
      checkpoint is selected by validation PR-AUC rather than the last
      epoch, and test metrics are reported from that checkpoint
- [x] Run both from-scratch baselines (SAGE, GAT) to convergence with early
      stopping (patience 10, tracked on val PR-AUC) and save a results table
      — **done, including a full round of bug-fixing along the way.** Bugs
      found and fixed in sequence:
      - `load_config()` was silently discarding YAML config values in favor
        of CLI argparse defaults — fixed.
      - A run labeled as GAT (Version 5) was accidentally training
        `GraphSAGEModel` the whole time (near-identical loss/metric numbers
        to the real SAGE run gave it away); root-caused and the real GAT run
        redone.
      - GAT training was unstable on its first real run (sharp single-epoch
        collapses, PR-AUC swinging 0.148→0.049, never beating SAGE) — fixed
        via gradient clipping + a lowered learning rate (`lr: 0.001`).
      - A NaN-imputation bug was found in `build_graph.py`'s feature
        pipeline, requiring a full `graph.pt` rebuild and baseline rerun.

      **Results table (best checkpoint per run, selected by val PR-AUC):**

      | Run | Best PR-AUC | Best ROC-AUC | Best epoch | Stopped at |
      |---|---|---|---|---|
      | SAGE (original graph) | 0.4106 | 0.8591 | 6 | 16 |
      | GAT (original graph, unstable — superseded) | 0.1478 | ~0.74 (never stable) | 21 | 40 |
      | GAT (grad-clip + lower-LR fix, original graph) | 0.3157 | 0.8103 | 6 | 16 |
      | SAGE (rebuilt graph, NaN-imputation fix) | **0.4147** | 0.8618 | 15 | 25 |
      | GAT (rebuilt graph, NaN-imputation fix) | **not yet run** | — | — | — |

      SAGE currently beats GAT on PR-AUC in every valid comparison so far.
      GAT has not yet been rerun against the NaN-imputation-fixed graph —
      **this is the actual next open item**, not a completed rerun (an
      earlier status-report pass in this project incorrectly said both were
      "rerun once each against the corrected graph"; only SAGE actually was
      — correcting that here).
- [ ] **Train one non-graph reference baseline** — LightGBM/XGBoost (or, as a
      cheaper fallback, plain logistic regression) on the node feature matrix
      alone, completely ignoring graph structure, with the same splits and
      imbalance handling as above. This is the comparison that actually tells
      you whether the graph is earning its added complexity — without it, you
      can only ever compare GNN variants against each other and can never
      answer "did the graph help at all?" It's also cheap relative to
      everything else here (hours, not days, since you're not hand-coding it).
      Don't be surprised if it's competitive: IEEE-CIS is a Kaggle dataset
      where tree-based models have historically scored very well, and
      GADBench-style benchmarks routinely include a tabular baseline for
      exactly this reason. If it wins, that's still a real, honestly-reportable
      finding — it reframes your report's contribution toward "here's
      specifically where/why graph structure and camouflage-resistance help"
      rather than "graphs beat tables," which is more defensible either way.
      **Not started.**
- [ ] **Run test-set evaluation via `evaluate.py`.** A checkpoint-unwrapping
      bug was found in `evaluate.py` (it wasn't unwrapping the `{'model':
      ...}` checkpoint dict before `load_state_dict()`) — meaning test-set
      evaluation was completely non-functional until that fix. The bug has
      been fixed, but evaluation **has still never actually been run, even
      once, on any checkpoint** — every PR-AUC/ROC-AUC number recorded above
      is a *validation*-set number from training, not a held-out test-set
      number. This is a real gap: the project currently has no confirmed
      test-set results at all.

Concepts to understand: why accuracy misleads under this level of class
imbalance (see Phase 1 for the exact figures); early-stopping on PR-AUC
rather than raw loss; why tree-based models are historically hard to beat on
tabular fraud data, and what that does and doesn't tell you about whether
relational structure matters.

Agent guardrails: every run's config and metrics get logged under
`experiments/<run-name>/` — nothing lives only in terminal output. Record the
seed used for each run. Also checkpoint model weights every few epochs (or
every N minutes) and commit the notebook version well before your session hits
the 12-hour wall — losing a multi-hour run to a timeout is the single most
avoidable way to burn your weekly GPU quota. Log wall-clock time per run too;
Phase 8's heavier sweep will need it.

Definition of Done: a checked-in results table (model, PR-AUC, ROC-AUC, F1,
recall) for both from-scratch baselines **and the tabular reference
baseline**, confirmed on the **test** split via `evaluate.py`, not just
validation numbers from training. **Not yet met** — SAGE and GAT both have
validation-set results checked into `experiments/`, but GAT hasn't been
rerun against the corrected graph, the tabular baseline hasn't been started,
and no test-set evaluation has been run at all (F1/recall are also not yet
tracked anywhere — only PR-AUC/ROC-AUC have been logged so far).

---

## Phase 6 — Literature deep-dive: camouflage-resistant GNNs
**~4–5 days, can run in parallel with Phase 4–5**

Goal: understand 3–4 papers' actual mechanisms well enough to explain them
without notes. This is what makes Phase 7 a real contribution instead of a
reskin of someone else's idea.

Required reading (write a ≤1-page mechanism summary per paper, in your own
words — this becomes part of your report's related-work section):
- **CARE-GNN** (Dou et al., CIKM 2020) — filters which neighbors get aggregated
  per relation using a label-aware similarity measure, with the similarity
  threshold adapted during training via a reinforcement-learning module
- **PC-GNN** (Liu et al., WWW 2021) — a node-level resampler ("pick and choose")
  combined with a label-aware neighbor selector, aimed at the class-imbalance
  side specifically
- **RL-GNN** (Scientific Reports, 2025) — required, not optional. This is the
  closest existing work to your Phase 7 (GAT + RL controller, evaluated on
  IEEE-CIS directly, 0.872 AUROC). You need to be able to state precisely, in
  one paragraph, how your approach differs from this specific paper
- One more paper from Appendix D's "recent camouflage-specific work" list
  (PROD or SCFCRC are good picks — both explicitly target the combined
  feature-camouflage + relation-camouflage problem, close to your Phase 7
  framing)

Tasks
- [ ] Write the four mechanism summaries and commit them — CARE-GNN's actual
      label-aware filtering mechanism has already been studied and applied
      correctly in Phase 7's design (see below), but none of the four
      summaries has been written up and committed as a standalone document
      yet
- [ ] Write one clear paragraph stating exactly what you will do **differently**
      — a simplified or modified selection rule, a different similarity measure,
      combining ideas from two papers, the specific from-scratch/ablation angle
      — anything specific and defensible. "Applying this to transaction data"
      alone is *not* a valid answer (see the novelty note in §1) — be precise
      about what's actually new. The concrete answer for this project: a
      from-scratch GAT baseline with a label-aware neighbor-trust signal
      (per-node predicted fraud scores, with trust between neighbors based on
      score agreement — see Phase 7), evaluated on IEEE-CIS with a direct
      comparison against RL-GNN's published 0.872 AUROC. This still needs to
      be written down as its own committed paragraph.

Agent guardrails: this phase produces prose notes, not code. If asked to
"implement CARE-GNN," push back and confirm scope with the user first — Phase 7
must be an original variant, not a port of a reference repo.

Definition of Done: four mechanism summaries committed; one paragraph on your
specific proposed twist, checked against "is this actually different, and can I
defend that in five minutes to my professor."

---

## Phase 7 — Design & implement the novel module
**~2–2.5 weeks**

Goal: your actual contribution.

Recommended default direction: a camouflage-resistant neighbor-selection
mechanism layered on top of your Phase 4 GAT, applied to IEEE-CIS. CARE-GNN and
PC-GNN's original papers validate on review-fraud graphs (Yelp/Amazon); the
defensible novelty claim here isn't "first on transaction data" (it isn't —
T-Finance/T-Social/S-FFSD already cover that ground, see Appendix D) but the
specific combination you're running: an explicit, ablated neighbor-filtering
mechanism, benchmarked against from-scratch GraphSAGE/GAT baselines you built
yourself, on IEEE-CIS specifically, with a direct comparison point against the
2025 GAT+RL result (Appendix D — RL-GNN). Carry this exact framing into your
report's contribution statement — it's precise and it holds up against a
literature-aware reader.

**Chosen mechanism: label-aware similarity-gated attention.** Each
camouflage-GAT layer (`camouflage_gat.py`) has a small per-node `score_head`
(`nn.Linear`) that predicts a node's own fraud-likelihood from its projected
features alone (no graph). Trust between two connected nodes is then
`exp(-|score_src - score_dst|)` — neighbors are trusted when their
independently-predicted fraud scores *agree*, not when their raw features
look similar. This is a deliberate, corrected design choice: an earlier
version of this mechanism used raw feature cosine similarity as the trust
signal, but that was identified as backwards for a camouflage-resistance
goal specifically — a camouflaged fraud node's entire purpose is to *look*
similar to normal nodes in raw features, so rewarding feature similarity
rewards successful camouflage rather than resisting it. The label-aware
version is a closer, correct match to CARE-GNN's actual per-relation,
label-aware filtering idea, adapted as an original mechanism rather than a
port of CARE-GNN's code.

Tasks
- [x] Implement the chosen mechanism as a module wrapping/extending the
      Phase 4 GAT — done: `camouflage_gat.py`'s `score_head` and label-aware
      trust signal, with `FraudCamouflageGNN` in `models.py` exposing
      `auxiliary_node_scores()` (the cached per-node scores, averaged across
      heads and layers) so the scores can be supervised directly. `train.py`
      adds a supervised auxiliary BCE loss against the true fraud labels,
      weighted by a new `aux_loss_weight` config value (default `0.3`,
      not yet tuned), alongside the main classification loss — this
      supervision is what makes the trust signal meaningful in the first
      place, since an untrained `score_head` would produce meaningless
      "agreement" scores.
- [ ] Build a simpler fixed-threshold/fixed-rule fallback version of the
      same trust signal (no learned `score_head`) as a lower-risk backup —
      not currently built. The implementation went straight to the learned
      version; having a simple fallback ready is worth doing before heavy
      Kaggle iteration starts, in case the learned version has convergence
      trouble.
- [ ] Get it training end-to-end on a small subsample first for fast
      iteration, then on the full graph — **attempted on the full graph, but
      collapsed, not yet successfully trained.** The camouflage-GAT run that
      actually completed showed val ROC-AUC sitting at ~0.5 (random-chance
      performance, e.g. 0.4985 by epoch 8) and PR-AUC bouncing as pure noise
      (0.0856 → 0.0911 → 0.0405 → 0.0596 → ...), with no recovery — the run
      was cancelled rather than let finish. This directly motivated running
      SAGE and GAT first as a sanity check: since **both plain baselines
      train normally** (SAGE reaching PR-AUC 0.41+, GAT 0.31+ once fixed),
      the bug is isolated to the camouflage mechanism itself, not the shared
      data/training pipeline. Leading suspects, per the diagnostic already
      laid out during baseline debugging: the untuned `aux_loss_weight`
      (`0.3`, never validated) overwhelming the main classification loss, or
      a bug in `CamouflageGATConv`'s trust-signal computation actively
      hurting rather than helping attention. **Neither has actually been
      diagnosed or fixed yet** — this is open work, not resolved by the
      baseline debugging that ruled out the pipeline as the cause.
- [ ] Compare against the Phase 5 baseline numbers on the **same split and seed**
      — blocked on this phase's training run actually working (see above)
      and on Phase 5's GAT-rerun and tabular-baseline gaps closing first.

Agent guardrails: keep the mechanism swappable behind a config flag so Phase 8's
ablations are config changes, not code forks. This is the most iteration-heavy
phase in the whole project — do all correctness debugging on the local
subsample, and only move to Kaggle once a version is confirmed stable. Never
treat a Kaggle session as the place to iterate on a new idea.

Definition of Done: trains stably; is at least directionally comparable to the
baseline on PR-AUC. If it's worse, that is still a valid, reportable result as
long as you can explain why — flag this to the user rather than quietly tuning
until the number looks better. **Not yet met** — the model has not yet
trained stably at all (collapsed to random-chance ROC-AUC on its only
completed full-graph run); diagnosing and fixing that collapse is the
immediate blocker for this entire phase.

---

## Phase 8 — Experiments & ablations
**~1–1.5 weeks**

Goal: turn one result into a defensible set of experiments.

Tasks
- [ ] Main comparison table: GraphSAGE, GAT, GAT + your module
- [ ] Ablation: your module with each key component removed, one at a time
- [ ] Run 3 random seeds per config; report mean ± std, never a single run
- [ ] Report the per-seed values in an appendix table too, not just mean ± std
      — with only 3 seeds, an honest reader will want to see whether your
      module wins in all 3 individually or only on average. If the margin over
      baseline is small, say so plainly in the report rather than letting the
      mean imply more consistency than 3 runs can support
- [ ] Note RL-GNN's published 0.872 AUROC / 0.683 AP (Appendix D) alongside
      your table as an external reference point — not a strict
      apples-to-apples comparison (different splits/preprocessing almost
      certainly), but useful context, and expect your professor or a
      reviewer to ask how you compare to it
- [ ] **Synthetic camouflage stress test.** Take your held-out known-fraud test
      nodes and synthetically add extra edges from a subset of them to random
      benign nodes, in steps (e.g. +0, +5, +10, +20 edges per fraud node), then
      re-run inference at each step — no retraining needed — and plot PR-AUC
      (or mean fraud-score for that subset) against injection level, one line
      per model (GraphSAGE, GAT, GAT + your module). This is the experiment
      that actually tests the *"camouflage-resistant"* claim in the project
      title: everything else here measures performance at the one, fixed
      camouflage level fraudsters already baked into the dataset, but never
      checks whether your mechanism holds up as camouflage gets *worse*. It's
      cheap (no retraining, just edge injection + forward passes at eval time),
      and a curve where your module degrades more slowly than the baselines is
      a substantially stronger headline result than a single-point PR-AUC delta.

Agent guardrails: never hand-pick the best-looking seed as the headline number —
report the aggregate across seeds. Before launching the full ablation × seed ×
stress-test grid, multiply it out against the per-run time you logged in
Phase 5/7 — if the total clears one week's 30 GPU-hours, trim the grid (fewer
seeds, fewer injection steps) up front rather than discovering the shortfall
mid-week.

Definition of Done: results table plus 1–2 figures (a PR curve, or an ablation
bar chart) **and the camouflage-degradation curve from the stress test above**
saved to `report/figures/`.

---

## Phase 9 — Error analysis (stretch goal)
**~3–5 days**

Goal: a qualitative story for your report/defense — *why* it works, not just
*that* it works.

Tasks
- [ ] Find cases the baseline got wrong that your module fixed, and vice versa
- [ ] Slice the "baseline got wrong, module fixed" cases by `TransactionAmt`
      (e.g. top vs. bottom quartile) to check whether your module's gains
      concentrate on high-value camouflaged fraud specifically. A finding
      like "this mostly catches large, well-disguised transactions the
      baseline missed" is a much stronger qualitative story for your
      report/defense than an aggregate PR-AUC delta — and if the gains are
      spread evenly instead, that's a fine, honest thing to report too
- [ ] Inspect attention weights on a handful of known-fraud nodes, before vs.
      after your module
- [ ] Optional: a simple explanation output — e.g. the top-k neighbors or
      relations that most influenced a flagged node's score

Definition of Done: 3–5 concrete examples, each with a short written explanation.

---

## Phase 10 — Report, reproducibility & final packaging
**~1.5–2 weeks, overlapping with Phase 8–9**

Tasks
- [ ] Write the report: motivation, related work (from Phase 6), method,
      experiments (Phase 8), results, limitations, honest discussion of what
      didn't work
- [ ] Clean the repo: final `README.md` with exact run commands, pinned
      `requirements.txt`, dead notebooks/code removed
- [ ] **Confirm a fresh clone + fresh environment reproduces your headline
      number.** This is the single most common thing that quietly breaks.
- [ ] Add a minimal CI workflow (`.github/workflows/test.yml`) that runs your
      Phase 3/4/7 unit tests on every push — a dozen lines of YAML (checkout,
      set up Python, `pip install -r requirements.txt`, `pytest tests/`). This
      turns your own top-level Definition of Done ("a codebase a stranger
      could clone and reproduce your headline number from") from a one-time
      manual check at the end into something enforced from the day you write
      your first unit test in Phase 3 — and it's a concrete thing to point to
      if your professor asks about engineering rigor
- [ ] Prepare a short talking-point summary for professor discussion — e.g.
      *"Normal fraud-detection AI gets fooled when criminals deliberately make
      themselves look normal — my project builds one that's harder to fool."*

Definition of Done: fresh-clone reproducibility check passes; report draft
complete; repo tagged (e.g. `v1.0-submission`).

---

## Appendix A — Dataset decision matrix

| Factor | IEEE-CIS | Elliptic |
|---|---|---|
| Graph readiness | Tabular — you build the graph (more work, more learning) | Already a graph (nodes/edges provided) |
| Size | ~590K transactions | 203,769 nodes, 234,355 edges |
| Features | Mixed transaction + identity fields, engineered "V" features | 166 numeric features (94 local + 72 neighbor-aggregate, already computed) — sources disagree by one (165 vs. 166); confirm with `print(data.x.shape)` once loaded rather than trusting either number |
| Labels | Binary `isFraud`, ~3–4% positive | 3-way (licit/illicit/unknown); ~2% illicit, 21% licit, 77% unknown |
| Split strategy | Time-based recommended | Time-based required (49 sequential steps) |
| Fit for the camouflage angle | Strong — shared card/device/email is a direct camouflage signal | Weaker — features are anonymized aggregates, less of an obvious "disguise" story |
| Main risk | Graph construction (hub nodes, relation design) eats your timeline | Structure-vs-temporal-shift confound (Appendix D) makes some GNN gains hard to attribute cleanly |

## Appendix B — Timeline (12-week default)

Adjust to your actual deadline — compress by dropping Phase 9, or stretch Phase 7
if your novel mechanism needs more iteration.

| Week | Phase(s) | Milestone |
|---|---|---|
| 1 | 0, 1 | Env ready, dataset locked, problem statement written |
| 2 | 2 | Graph construction script done |
| 3 | 3 | GraphSAGE from scratch, tested |
| 4 | 4, 6 (start) | GAT from scratch, tested; reading started |
| 5 | 5, 6 (finish) | Baseline results table; mechanism summaries done |
| 6–7 | 7 | Novel module implemented, training end-to-end |
| 8 | 7 (finish), 8 (start) | Novel module beats/matches baseline directionally |
| 9 | 8 | Ablations + multi-seed results done |
| 10 | 9 | Error analysis examples collected |
| 11–12 | 10 | Report written, repro check passed, submission packaged |

*The tabular baseline (Phase 5), the stress-test ablation (Phase 8), and the CI
workflow (Phase 10) are each on the order of a few hours, not days — they
should fit inside the existing per-phase estimates above without pushing the
12-week total.*

*The week estimates above are wall-clock, not GPU-hours — but Phases 3, 4, 5,
7, and 8 are exactly the ones drawing on your 30 GPU-hr/week quota. If a phase
needs more GPU time than a week gives you, it spills into next week's
allowance; plan Phases 7–8 especially with this in mind rather than assuming a
week's estimate and a week's quota line up on their own.*

## Appendix C — "From scratch" scope clarification

**Allowed:** `torch.nn.Linear`, autograd, optimizers, `torch.sparse`, raw tensor
indexing, native PyTorch scatter/reduce ops (`scatter_reduce_`, `index_add_`) or
`torch_scatter`'s reduction functions as a fallback (see §3 — both are
primitive index-reduce operations, not models), and PyTorch Geometric's
`Data`/`Dataset` classes purely for loading/storing graphs.

**Not allowed:** `torch_geometric.nn.SAGEConv`, `GATConv`, or any other
prebuilt message-passing layer from PyG, DGL, or similar libraries — for either
the baselines or the novel module.

This line (data-loading utilities are fine, model layers are not) is a
reasonable reading of "no pretrained models," but it's still worth a one-line
confirmation from your professor early on, since "no pretrained models" is
slightly ambiguous about utility functions like this.

## Appendix D — Reading list (for citation, not for copying)

**Foundational methods (original camouflage-resistant GNNs):**
- CARE-GNN — Dou et al., *Enhancing Graph Neural Network-based Fraud Detectors
  against Camouflaged Fraudsters*, CIKM 2020. Code: github.com/YingtongDou/CARE-GNN
- PC-GNN — Liu et al., *Pick and Choose: A GNN-based Imbalanced Learning
  Approach for Fraud Detection*, WWW 2021
- H2-FDetector — Shi et al., *H2-FDetector: A GNN-based Fraud Detector with
  Homophilic and Heterophilic Connections*, WWW 2022 — separate aggregation
  strategies for homophilic vs. heterophilic connections
- GAGA — Wang et al., *Label Information Enhanced Fraud Detection against Low
  Homophily in Graphs*, WWW 2023 — group aggregation for distinguishable
  multi-hop neighborhood information

**Transaction-graph benchmarks and closest related work — read these before
finalizing your novelty paragraph, they directly constrain what you can claim:**
- T-Finance / T-Social — Tang et al., *Rethinking Graph Neural Networks for
  Anomaly Detection*, ICML 2022 — transaction/account-graph fraud benchmarks;
  establishes that "transaction data" alone is not the open gap. Introduces
  BWGNN plus the T-Finance/T-Social datasets.
- S-FFSD — Xiang et al., *Semi-supervised Credit Card Fraud Detection via
  Attribute-driven Graph Representation*, AAAI 2023 — simulated credit-card
  transaction graph, same purpose as T-Finance. The original paper calls the
  dataset FFSD; "S-FFSD" is the name commonly used for the publicly-released
  version in follow-up work.
- **RL-GNN** — *Reinforcement learning with graph neural network (RL-GNN)
  fusion for real-time financial fraud detection*, Scientific Reports, Dec
  2025 — GAT + RL controller evaluated directly on IEEE-CIS, 0.872 AUROC /
  0.683 AP. The closest existing work to Phase 7 — required reading (Phase 6),
  and the paper you need to explicitly differentiate from in your report.
  Devi, Raja & Chin, *Sci Rep* 15, 42953 (2025),
  DOI 10.1038/s41598-025-25200-3.
- **Naming heads-up:** don't confuse RL-GNN above with a second, similarly-named
  2025 paper, *FraudGNN-RL* (Cui et al., IEEE Open Journal of the Computer
  Society, 2025, DOI 10.1109/OJCS.2025.3543450). It's also a GNN+RL fraud
  framework but a genuinely different method (a Temporal-Spatial-Semantic
  Graph Convolution architecture with a DQN that adjusts thresholds),
  evaluated separately. Easy to conflate the two in a lit review — worth a
  one-line disambiguation in your report if you cite either.
- GADBench — benchmark paper standardizing evaluation of CARE-GNN, PC-GNN, and
  related methods; useful for baseline-comparison methodology

**Recent camouflage-specific work (2025) — for currency in your related-work
section:**
- PROD — *Projected and Orthogonal Disentanglement*, Knowledge-Based Systems —
  tackles scarce labeled data and camouflage jointly via risk-aware encoding
  and disentanglement. The publisher lists this in Volume 343 (2026), not
  2025 — likely an online-first-vs-print-volume gap; cite the DOI rather than
  a year if a reference manager pushes back.
- SCFCRC — *Simultaneously Counteract Feature Camouflage and Relation
  Camouflage for Fraud Detection*, arXiv 2025 — directly targets both
  camouflage types together via a Feature Camouflage Filter and a Relation
  Camouflage Refiner, same framing as your Phase 7 problem statement.
  arXiv:2501.12430, Zhang, Ye, Zhao, Wang & Su.
- The underlying critique that per-relation neighbor selectors handle relation
  camouflage well but degrade once feature camouflage is layered on top too is
  real and current — it's SCFCRC (above) that makes this exact argument
  explicitly and recently. HA-GNN (arXiv:2202.06096, *Improving Fraud
  Detection via Hierarchical Attention-based Graph Neural Network*) covers
  related ground but is a 2022 paper — cite it with that date if you use it,
  and prefer SCFCRC for the 2025-current framing of this specific point.
- **GNN-LAARA** — *Fraud detection based on GNNs with local augmentation and
  adaptive relation aggregation*, Expert Systems with Applications, Oct
  2025 — combines CVAE-based feature enhancement, DDPG (RL)-based adaptive
  neighbor selection, and multi-relational attention to counter feature and
  relation camouflage together. Closely adjacent to both your Phase 7
  mechanism menu and the RL-GNN reading above — worth a skim, and a good third
  data point alongside RL-GNN and SCFCRC for your "here's exactly how mine
  differs" paragraph

**Datasets:**
- IEEE-CIS Fraud Detection dataset — Kaggle
- Elliptic Bitcoin dataset — Kaggle, or `torch_geometric.datasets.EllipticBitcoinDataset`

## Appendix E — Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Graph construction (Phase 2) takes longer than a week | Medium | Time-box it; fall back to Elliptic if you're not done by end of week 2 |
| Novel module (Phase 7) doesn't beat baseline | Medium | Still a valid, explainable result — budget time to analyze *why*, don't just keep tuning |
| Novel module (Phase 7) fails to train at all (not just "underperforms") | **Realized** — camouflage-GAT's only full-graph run collapsed to ROC-AUC ≈ 0.5 | Ruled out the shared pipeline as the cause by confirming both baselines train normally first; remaining suspects are `aux_loss_weight` tuning and the trust-signal computation in `CamouflageGATConv` — diagnose there next, starting with a subsample run at a much smaller `aux_loss_weight` before returning to the full graph |
| A harness bug silently produces wrong or no results without erroring | **Realized twice** — `evaluate.py` never unwrapped the checkpoint dict, so test-set evaluation silently produced nothing until caught; a training run also silently trained the wrong model class (SAGE instead of GAT) due to a `load_config()` bug | Treat "the script ran without an error" as insufficient evidence of correctness — spot-check a completed run's actual saved config/output against what was intended, not just its exit code |
| Hub-node explosion makes the graph unusable | Medium–High (IEEE-CIS) | Degree caps from Phase 2, checked immediately after construction, not discovered mid-training |
| Running out of time for the report | High if left until the end | Start the related-work section in Phase 6, not Phase 10 |
| "From scratch" scope dispute with professor | Low, but costly if it happens | Confirm Appendix C's line with them in week 1 |
| Novelty claim challenged as "already done" (T-Finance/S-FFSD/RL-GNN exist) | Low, with mitigation in place | Precise novelty statement in §1 and Phase 7; RL-GNN is required reading in Phase 6 so the differentiation paragraph is specific, not naive |
| Tabular baseline (Phase 5) matches or beats every graph model | Medium | Still a valid, honestly-reportable finding — reframe the report's contribution around *when/why* graph structure and camouflage-resistance help rather than *whether* graphs beat tables; the Phase 8 camouflage-specific ablations stay meaningful either way |
| Synthetic camouflage injection (Phase 8) doesn't move any model's score | Medium | Also a valid, reportable result — but first check the injection isn't so large it saturates every model's neighborhood indiscriminately; sweep several injection levels before concluding the mechanism doesn't matter |
| Weekly GPU quota (30h) or 12h session cap runs out mid-phase | High during Phases 7–8 | Checkpoint/resume every run (Phase 0); do EDA, graph construction, and unit-testing on CPU-only sessions, which don't draw down GPU quota; budget Phase 8's full sweep against the weekly cap before launching it |
| Kaggle session disconnects or idles out, losing an unsaved run | Medium | Same checkpoint/resume discipline as above; commit ("Save Version") after every meaningful run rather than relying on an interactive session's live state |
| Locally-authored code (Antigravity) behaves differently on Kaggle's pre-built image (library version drift) | Medium | Pin versions in `requirements.txt` against what Kaggle's image actually has (Phase 0); run a trivial push-run-pull dry cycle before Phase 1 rather than discovering mismatches under time pressure |