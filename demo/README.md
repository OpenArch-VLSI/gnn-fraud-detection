# Demo: LLM explanation of a flagged transaction (not working yet)

`agent.py` reads a flagged transaction's links and the model's attention to them, then asks an LLM to write a short plain-English note. It is not part of the fraud results.

Known problems before it can run:
1. PyG attention layers don't store attention weights by default. The model needs an option to return them.
2. It loads `experiments/camo_gat_baseline/best_model.pt`, which exists only after the camouflage model is trained.
3. The model name in `agent.py` must be checked against Google's current list.
4. It sends neighbourhood data and fraud labels to an external API. Decide whether that is acceptable first.

Setup (later): `python -m pip install -r demo/requirements-demo.txt`, and put `GOOGLE_API_KEY` in a `.env` file in the project root.