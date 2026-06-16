# Results Summary

## Runtime

DCEE balanced row: mean 81.85 ms/slot, p95 212.70 ms/slot, 10.03 solves/slot, and 100% solver success.

## FC-Union Benchmark

The final FC-Union benchmark uses deterministic rescue; after-rescue failed candidate evaluations are zero, benchmark warning rows are zero, and solver success after rescue is 100%.

## Formal DRL Baselines

Held-out test: 10 seeds and 200 slots per seed. All methods use the same exact post-motion resource solver.

- DCEE-default: penalty 2.202 +/- 0.029, backlog 218.902 +/- 39.804
- DCEE-tuned: penalty 2.049 +/- 0.119, backlog 169.156 +/- 82.008
- DDPG-Mobility-Exact: penalty 2.134 +/- 0.031, backlog 522.132 +/- 86.564
- PPO-Mobility-Exact: penalty 2.113 +/- 0.023, backlog 500.016 +/- 119.786
- SAC-Mobility-Exact: penalty 2.107 +/- 0.039, backlog 523.262 +/- 76.367

DCEE-tuned backlog reduction is approximately 66.2% versus PPO, 67.7% versus SAC, and 67.6% versus DDPG.

## Theory and Robustness

The balanced profile is a stress profile. The relaxed profile is a positive-slack sanity check for AoI and uncertainty margins. Channel/primitive experiments are model-sensitivity diagnostics, not field validation.

## Altitude

The compact altitude experiment supports the formal altitude-extension interface, but it is not a full 3D deployment optimization study.
