# Reproduction Guide

## Tier 1: Quick Verification

```bash
pip install -r requirements.txt
python -m pytest tests_r1
python scripts_r1/run_r1_smoke.py --slots 2 --seeds 1 --controllers proposed_full,static_uav
```

## Tier 2: Reproduce Tables/Figures from Included Summaries

Final processed CSV files are under `data/processed/`. The final table and figure outputs are already included in `tables/` and `figures/`.

## Tier 3: Full Experimental Reproduction

Representative commands:

```bash
python scripts_r1/run_r1_dcee_tuning_formal.py --config configs_r1/formal_dcee_tuning.json --outdir results_r1/formal_drl_tuning
python scripts_r1/run_r1_formal_drl_training.py --config configs_r1/formal_drl_training.json --outdir results_r1/formal_drl_tuning
python scripts_r1/evaluate_r1_formal_drl.py --config configs_r1/formal_drl_training.json --outdir results_r1/formal_drl_tuning --dcee-config results_r1/formal_drl_tuning/best_dcee_config_formal.json --selected-checkpoints results_r1/formal_drl_tuning/selected_drl_checkpoints.json
python scripts_r1/analyze_formal_drl_results.py --root-dir results_r1/formal_drl_tuning --tables-dir tables --figures-dir figures
```

Full reproduction can take hours. Large raw candidate and slot CSVs are not shipped but can be regenerated.
