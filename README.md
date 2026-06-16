# Decision-Consistent Online Control for Move-Then-Serve UAV-Enabled ISAC

This repository contains the clean R1 experiment code and processed artifacts for **Decision-Consistent Online Control for Move-Then-Serve UAV-Enabled Integrated Sensing and Communication**.

## What This Repository Contains

R1 simulator and controller code, final experiment runners, final configs, processed summaries, manuscript-ready tables, manuscript-ready figures, and reproducibility documentation.

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

## Expected Runtime

Quick tests take minutes. Full FC-Union and formal DRL reproduction can take hours on a local CPU.

## How to Cite

See `CITATION.cff`.

## License

Source code is released under the MIT License. Processed summaries, generated
tables, and generated figures are released under CC BY 4.0; see
`DATA_LICENSE.md`.

## Contact

Weilin Chang, weilin.zju@gmail.com
