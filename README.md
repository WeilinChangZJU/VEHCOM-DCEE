# Decision-Consistent Online Control for Move-Then-Serve UAV-Enabled ISAC

## What This Repository Contains

Simulator and controller code, final experiment runners, final configs, processed summaries, manuscript-ready tables, manuscript-ready figures, and reproducibility documentation.

## Key Claims Supported by Artifacts

- The FC-Union finite-family benchmark uses a union candidate family and deterministic rescue.
- DCEE has main-performance, runtime, queue/slack, model-robustness, learning-baseline, and altitude-sensitivity diagnostics.
- Validation-tuned DCEE outperforms representative PPO/SAC/DDPG mobility baselines on the held-out penalty metric in the included formal learning-baseline study and substantially reduces backlog.
- Physical/channel/primitive results are model-sensitivity diagnostics, not measured field validation.

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python -m pytest tests_r1
python scripts_r1/run_r1_smoke.py --slots 2 --seeds 1 --controllers proposed_full,static_uav
```

## Reproducing Tables and Figures

See `REPRODUCE.md`. Processed summaries are under `data/processed/`, tables under `tables/`, and figures under `figures/`.

## Experiment Groups

Main performance; FC-Union benchmark; runtime; queue/Slater/V diagnostics; physical/channel/primitive robustness; formal DRL baselines; altitude sensitivity.
