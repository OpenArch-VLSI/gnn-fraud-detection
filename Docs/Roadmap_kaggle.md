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

## 2. Top-level definition of done

- [ ] From-scratch GraphSAGE and GAT baselines, trained and evaluated on a real
      fraud dataset, with imbalance-aware metrics (not accuracy)
- [ ] A non-graph reference baseline, so "the graph helped" is a measured
      result and not an assumption
- [ ] At least one experiment testing robustness to *escalating* camouflage,
      not just performance at the dataset's fixed, natural camouflage level
- [ ] One clearly-scoped novel extension, implemented, ablated, and compared
      fairly against the baselines on identical splits/seeds
- [ ] A codebase a stranger could clone and reproduce your headline number from
- [ ] A written report: motivation, related work, method, results, honest
      limitations
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

**Decision point — IEEE-CIS vs. Elliptic** (full comparison in Appendix A).
Default recommendation: **IEEE-CIS**, because constructing the multi-relation
graph yourself is real learning *and* sets up the camouflage angle in Phase 7
naturally — a fraud ring sharing a device or card is a direct camouflage signal.
Elliptic is already a graph (less construction work, less to learn there) and is a
reasonable fallback if graph construction eats too much of your timeline.

Tasks
- [ ] Attach the chosen dataset as a notebook input (Kaggle already hosts
      both — search for the IEEE-CIS Fraud Detection competition or the
      Elliptic Bitcoin dataset and "Add Input"; join the competition's rules
      first if prompted, it's free) rather than downloading it anywhere
      yourself; Elliptic is also available via
      `torch_geometric.datasets.EllipticBitcoinDataset` if you'd rather fetch it
      in code
- [ ] Do this phase's EDA on a **CPU-only** Kaggle session (Accelerator → None)
      — it doesn't need a GPU, and CPU sessions don't draw down your 30-hour
      weekly GPU quota
- [ ] EDA notebook: class balance, missing values, feature types; for IEEE-CIS
      specifically, check the cardinality of candidate "shared entity" columns
      (`card1`–`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`,
      `DeviceInfo`)
- [ ] Write a one-paragraph problem statement and a one-paragraph novelty
      statement — this becomes tomorrow's pitch to your professor and later the
      intro of your report

Concepts to understand: why accuracy is a bad headline metric under class
imbalance — concretely, a model that always predicts "not fraud" already scores
~96.5% on IEEE-CIS (20,663 fraud / 590,540 total) and ~90.2% on Elliptic *if
evaluated on labeled nodes only* (4,545 illicit / 46,564 labeled — the higher
97%+ figure sometimes quoted only holds if you count the unlabeled 77% as
implicit negatives, which isn't standard practice and won't match how you'll
actually evaluate in Phase 5); why a **random** train/test split can leak
information in transaction data (better: time-based split, especially for
Elliptic's 49 timesteps).

Agent guardrails: this phase is analysis only — no training yet.

Definition of Done: EDA notebook committed; problem + novelty statements written;
dataset choice locked in `README.md`.

---

## Phase 2 — Graph construction pipeline
**~1 week**

Goal: a deterministic script that turns raw data into a graph object your models
can consume, run once, reused by every later phase.

**If IEEE-CIS:**
- [ ] Decide node scope: transactions as the only node type (simplest — shared
      entities become *relation types*, i.e. edges), vs. a fully heterogeneous
      graph with separate card/device/email nodes. Pick one and write down why —
      this is a real design decision, not busywork.
- [ ] Build one edge relation per shared-entity type (same card1+card2 → edge;
      same DeviceInfo → edge; same email domain → edge). This mirrors CARE-GNN's
      multi-relation design (it used relations like same-user / same-time /
      same-star on review data) — you're doing the same idea on transaction data.
- [ ] **Cap degree on hub entities.** A shared value like `gmail.com` will connect
      a huge fraction of all transactions if you don't cap it — this is a known,
      easy-to-miss gotcha that silently turns your graph into a near-clique.
- [ ] Serialize: per-relation `edge_index` tensors, node feature matrix, label
      vector, train/val/test masks → disk, so nothing downstream re-parses raw CSVs.

**If Elliptic:**
- [ ] Load the provided node features / edges / labels directly
- [ ] Build a **time-based** split (paper convention: train on early timesteps,
      test on later ones — e.g. steps 1–34 train, 35–49 test) — not a random split
- [ ] Decide how to handle the ~77% of nodes with unknown labels: drop them, or
      keep them for a semi-supervised setup — pick one and justify it

**Applies to either dataset:**
- [ ] Write one explicit sentence in `README.md` stating whether your setup is
      **transductive** (the full graph — including test-period nodes and edges
      — is visible at train time, and only labels are split by time) or
      **inductive** (test-period nodes are entirely absent from the graph
      during training). Given the time-based split above, most straightforward
      implementations end up transductive-with-temporal-label-masking, which is
      a legitimate, standard choice — it just needs to be a choice a reader can
      find stated plainly, not one they have to reverse-engineer from your code.

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
  in general. Kaggle's free GPUs are a small box (single P100/T4, 16GB VRAM,
  ~29GB system RAM once a GPU is attached), and PC-GNN's 128GB-RAM comparison
  point is ~4.5x more RAM than you actually have. Full-batch may still work on
  Elliptic (much smaller) but is genuinely uncertain on the full IEEE-CIS graph
  depending on your feature width and hidden size. Implement neighbor sampling
  for real — treat it as the likely default, profile full-batch memory on a
  small subsample first, and only skip sampling if that profiling says you can
  afford to
- The update rule: new embedding for node *v* = `σ(W · CONCAT(h_v, AGG({h_u for u in neighbors(v)})))`
- Why the "mean aggregator" is *not* the same as the GCN aggregator (different
  self-loop and normalization handling)

Tasks
- [ ] Implement a `SAGEConv` layer by hand using primitive tensor ops (see
      Appendix C for what "by hand" allows)
- [ ] Decide explicitly what your mean aggregator does with a zero-degree node
      (mean of an empty neighbor set is undefined) — a self-loop or a small
      learned "isolated-node" fallback vector are the two standard fixes.
      Double-check your Phase 2 degree caps don't quietly create zero-degree
      nodes you haven't accounted for.
- [ ] Stack two layers, add a binary classification head
- [ ] Unit test on a tiny synthetic 5-node graph: check output shape, and that
      `loss.backward()` runs cleanly with nonzero gradients

Agent guardrails: **do not** import `torch_geometric.nn.SAGEConv`,
`dgl.nn.SAGEConv`, or any prebuilt message-passing layer. Comment each line of
the layer with which part of the formula above it implements.

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
- [ ] Implement `GATConv` by hand — single head first, then extend to multi-head
- [ ] In your hand-written per-neighborhood softmax, subtract the max logit
      before exponentiating (the standard numerically-stable softmax trick). A
      naive `exp()` over raw, un-shifted attention logits is a common and
      easy-to-miss source of silent NaNs once neighborhood-size variance gets
      large near your Phase 2 degree-capped hub nodes. Apply the same
      zero-degree fallback you used in Phase 3.
- [ ] Unit test: attention weights sum to 1 across each node's neighborhood
- [ ] Sanity-visualize attention weights on a handful of nodes — are they
      spread out and meaningful, or collapsing to near-uniform?

Agent guardrails: same rule as Phase 3 — no `GATConv` import, ever.

Definition of Done: attention-sums-to-1 test passes; training is stable (loss
decreases, no NaNs).

---

## Phase 5 — Baseline training & evaluation harness
**~4–5.5 days**

Goal: a rigorous, reusable train/eval loop *before* touching the novel idea, so
Phase 7 has a trustworthy number to beat.

Tasks
- [ ] Handle class imbalance: class-weighted BCE loss at minimum; consider focal
      loss if weighting alone underperforms
- [ ] Metrics: **PR-AUC as the primary metric**, plus ROC-AUC, F1 at a chosen
      threshold, and recall at a fixed precision — accuracy is reported only as a
      footnote, never as the headline number
- [ ] Config-driven `train.py --config configs/sage_baseline.yaml`, logging
      metrics every epoch
- [ ] Run both baselines to convergence; save a results table
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
      rather than "graphs beat tables," which is more defensible either way

Concepts to understand: why accuracy misleads under this level of class
imbalance (see Phase 1 for the exact per-dataset figures); early-stopping on
PR-AUC rather than raw loss; why tree-based models are historically hard to
beat on tabular fraud data, and what that does and doesn't tell you about
whether relational structure matters.

Agent guardrails: every run's config and metrics get logged under
`experiments/<run-name>/` — nothing lives only in terminal output. Record the
seed used for each run. Also checkpoint model weights every few epochs (or
every N minutes) and commit the notebook version well before your session hits
the 12-hour wall — losing a multi-hour run to a timeout is the single most
avoidable way to burn your weekly GPU quota. Log wall-clock time per run too;
Phase 8's heavier sweep will need it.

Definition of Done: a checked-in results table (model, PR-AUC, ROC-AUC, F1,
recall) for both from-scratch baselines **and the tabular reference baseline**.

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
- [ ] Write the four mechanism summaries
- [ ] Write one clear paragraph stating exactly what you will do **differently**
      — a simplified or modified selection rule, a different similarity measure,
      combining ideas from two papers, the specific from-scratch/ablation angle
      — anything specific and defensible. "Applying this to transaction data"
      alone is *not* a valid answer (see the novelty note in §1) — be precise
      about what's actually new

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

Pick **one** of these starting mechanisms and adapt it — don't try to build all
three:
- [ ] **Similarity-gated attention** — before computing GAT attention, compute a
      feature-similarity score per node pair and down-weight or mask edges below
      a threshold (learned or heuristic)
- [ ] **Per-relation adaptive filtering** — for each relation type from Phase 2,
      learn a separate filtering rule (same spirit as CARE-GNN's per-relation
      similarity measure, but you define your own scoring function and update
      rule — don't copy theirs)
- [ ] **Label-aware contrastive term** — an auxiliary loss that pulls same-label
      neighbor embeddings together and pushes different-label pairs apart, making
      it structurally harder for a fraud node to hide inside a normal-looking
      neighborhood
      *(Caveat: if you're on Elliptic, remember 77% of nodes are unlabeled —
      this mechanism needs same/different-label neighbor pairs to form its
      contrastive terms, so your usable pool of pairs shrinks a lot on this
      dataset specifically. Not a blocker if you pick this option and Elliptic,
      just budget for it.)*

Tasks
- [ ] **Build this in two passes to de-risk the phase:**
  - [ ] **V0 — fixed-heuristic version**: implement your chosen mechanism with
        a hand-set threshold/rule instead of a learned one. Fast to build, and
        gives you a real, working comparison point within days, not weeks.
  - [ ] **V1 — learned version**: replace the fixed threshold/rule with the
        learned mechanism as scoped above.
  - [ ] If V1 has convergence trouble late in the timeline, V0 is still a
        legitimate, reportable data point for Phase 8 — a documented "the
        learned version didn't converge in time, here are the heuristic
        version's numbers instead" beats having nothing to show.
- [ ] Implement the chosen mechanism as a module wrapping/extending your Phase 4
      GAT
- [ ] Get it training end-to-end on a small subsample first for fast iteration,
      then on the full graph
- [ ] Compare against the Phase 5 baseline numbers on the **same split and seed**

Agent guardrails: keep the mechanism swappable behind a config flag so Phase 8's
ablations are config changes, not code forks. This is the most iteration-heavy
phase in the whole project — do all correctness debugging on the local
subsample, and only move to Kaggle once a version is confirmed stable. Never
treat a Kaggle session as the place to iterate on a new idea.

Definition of Done: trains stably; is at least directionally comparable to the
baseline on PR-AUC. If it's worse, that is still a valid, reportable result as
long as you can explain why — flag this to the user rather than quietly tuning
until the number looks better.

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
- [ ] **If using IEEE-CIS:** note RL-GNN's published 0.872 AUROC / 0.683 AP
      (Appendix D) alongside your table as an external reference point — not a
      strict apples-to-apples comparison (different splits/preprocessing almost
      certainly), but useful context, and expect your professor or a reviewer
      to ask how you compare to it
- [ ] **If using Elliptic:** explicitly check performance across the later,
      harder timesteps. Recent work has raised the concern that temporal
      distribution shift alone explains a meaningful chunk of apparent GNN gains
      on Elliptic — worth checking honestly rather than assuming the graph
      structure is doing all the work.
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
- [ ] **If using IEEE-CIS**, slice the "baseline got wrong, module fixed" cases
      by `TransactionAmt` (e.g. top vs. bottom quartile) to check whether your
      module's gains concentrate on high-value camouflaged fraud specifically.
      A finding like "this mostly catches large, well-disguised transactions
      the baseline missed" is a much stronger qualitative story for your
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
| Hub-node explosion makes the graph unusable | Medium–High (IEEE-CIS) | Degree caps from Phase 2, checked immediately after construction, not discovered mid-training |
| Running out of time for the report | High if left until the end | Start the related-work section in Phase 6, not Phase 10 |
| "From scratch" scope dispute with professor | Low, but costly if it happens | Confirm Appendix C's line with them in week 1 |
| Novelty claim challenged as "already done" (T-Finance/S-FFSD/RL-GNN exist) | Low, with mitigation in place | Precise novelty statement in §1 and Phase 7; RL-GNN is required reading in Phase 6 so the differentiation paragraph is specific, not naive |
| Tabular baseline (Phase 5) matches or beats every graph model | Medium | Still a valid, honestly-reportable finding — reframe the report's contribution around *when/why* graph structure and camouflage-resistance help rather than *whether* graphs beat tables; the Phase 8 camouflage-specific ablations stay meaningful either way |
| Synthetic camouflage injection (Phase 8) doesn't move any model's score | Medium | Also a valid, reportable result — but first check the injection isn't so large it saturates every model's neighborhood indiscriminately; sweep several injection levels before concluding the mechanism doesn't matter |
| Weekly GPU quota (30h) or 12h session cap runs out mid-phase | High during Phases 7–8 | Checkpoint/resume every run (Phase 0); do EDA, graph construction, and unit-testing on CPU-only sessions, which don't draw down GPU quota; budget Phase 8's full sweep against the weekly cap before launching it |
| Kaggle session disconnects or idles out, losing an unsaved run | Medium | Same checkpoint/resume discipline as above; commit ("Save Version") after every meaningful run rather than relying on an interactive session's live state |
| Locally-authored code (Antigravity) behaves differently on Kaggle's pre-built image (library version drift) | Medium | Pin versions in `requirements.txt` against what Kaggle's image actually has (Phase 0); run a trivial push-run-pull dry cycle before Phase 1 rather than discovering mismatches under time pressure |
