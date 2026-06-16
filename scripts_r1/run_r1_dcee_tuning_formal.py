from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_r1.run_r1_patch9b_performance_strengthening import (
    build_config,
    evaluate_method,
    scenario_config_to_dict,
    summarize_seed_rows,
)


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def make_variant_name(v: float, rho: float, shortlist: int) -> str:
    return f"V{int(v)}_rho{str(rho).replace('.', 'p')}_sl{int(shortlist)}"


def make_overrides(v: float, rho: float, shortlist: int) -> Dict[str, Any]:
    # Interpret "shortlist" as both the local directional shortlist and the
    # retained shortlist size. This keeps the tuning knob simple and auditable.
    return {
        "V": float(v),
        "rho_r": float(rho),
        "shortlist_num_local_dirs": int(shortlist),
        "shortlist_size": int(shortlist),
    }


def run_variant_set(
    base: Dict[str, Any],
    variants: List[Dict[str, Any]],
    seeds: Iterable[int],
    slots: int,
    outdir: Path,
    raw_name: str,
    slot_name: str,
    summary_name: str,
    log_handle,
) -> pd.DataFrame:
    summaries: List[pd.DataFrame] = []
    slot_rows: List[pd.DataFrame] = []
    for variant in variants:
        variant_name = str(variant["variant"])
        overrides = dict(variant.get("overrides", {}))
        cfg = build_config(base, overrides)
        part_summaries, part_slots = evaluate_method(
            cfg,
            method="proposed_full",
            label=f"DCEE-{variant_name}",
            seeds=seeds,
            slots=slots,
            variant=variant_name,
            log_handle=log_handle,
        )
        for df in part_summaries:
            df["overrides_json"] = json.dumps(overrides, sort_keys=True)
        summaries.extend(part_summaries)
        slot_rows.extend(part_slots)
    raw = pd.concat(summaries, ignore_index=True)
    slots_df = pd.concat(slot_rows, ignore_index=True)
    raw.to_csv(outdir / raw_name, index=False)
    slots_df.to_csv(outdir / slot_name, index=False)
    summary = summarize_seed_rows(raw, ["variant", "method", "method_label"])
    summary = summary.sort_values(["total_penalty_mean", "queue_backlog_mean"], ascending=[True, True])
    summary.to_csv(outdir / summary_name, index=False)
    return summary


def write_table(summary: pd.DataFrame, path: Path, title_comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Penalty & Backlog & Travel & Solver \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        solver = float(row.get("solver_success_mean", 0.0))
        rows.append(
            f"{row['variant']} & {float(row['total_penalty_mean']):.3f} $\\pm$ {float(row['total_penalty_std']):.3f} & "
            f"{float(row['queue_backlog_mean']):.3f} $\\pm$ {float(row['queue_backlog_std']):.3f} & "
            f"{float(row['travel_distance_mean']):.3f} $\\pm$ {float(row['travel_distance_std']):.3f} & "
            f"{100.0 * solver:.1f}\\% \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}", f"% {title_comment}"])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal DCEE validation tuning for Patch 10.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/formal_dcee_tuning.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/formal_drl_tuning"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables_r1"))
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--skip-stage2", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    cfg_json = read_config(args.config)
    outdir = args.outdir
    tables_dir = args.tables_dir
    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs_r1"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "patch10_dcee_tuning.log"

    base = cfg_json.get("base", {})
    stage1_seeds = [int(x) for x in cfg_json.get("stage1_validation_seeds", [20, 21, 22])]
    stage2_seeds = [int(x) for x in cfg_json.get("stage2_validation_seeds", [20, 21, 22, 23, 24])]
    test_seeds = [int(x) for x in cfg_json.get("test_seeds", list(range(100, 110)))]
    stage1_slots = int(cfg_json.get("stage1_slots", 100))
    stage2_slots = int(cfg_json.get("stage2_slots", 200))
    test_slots = int(cfg_json.get("test_slots", 200))
    top_k = int(cfg_json.get("stage2_top_k", 5))

    with log_path.open("w", encoding="utf-8") as log:
        t0 = time.perf_counter()
        emit("Patch 10 formal DCEE tuning", log)
        emit(f"Config: {args.config}", log)

        grid = cfg_json.get("stage1_grid", {})
        stage1_variants = [
            {
                "variant": make_variant_name(v, rho, sl),
                "overrides": make_overrides(v, rho, sl),
            }
            for v, rho, sl in itertools.product(grid.get("V", []), grid.get("rho_r", []), grid.get("shortlist", []))
        ]
        if not stage1_variants:
            raise ValueError("Stage-1 grid is empty.")

        if args.skip_stage1 and (outdir / "dcee_tuning_stage1.csv").exists():
            stage1_summary = pd.read_csv(outdir / "dcee_tuning_stage1.csv")
        else:
            emit(f"Stage 1: {len(stage1_variants)} configs, seeds={stage1_seeds}, slots={stage1_slots}", log)
            stage1_summary = run_variant_set(
                base,
                stage1_variants,
                stage1_seeds,
                stage1_slots,
                outdir,
                raw_name="dcee_tuning_stage1_raw.csv",
                slot_name="dcee_tuning_stage1_slot_metrics.csv",
                summary_name="dcee_tuning_stage1.csv",
                log_handle=log,
            )
        write_table(stage1_summary.head(10), tables_dir / "table_formal_dcee_tuning_stage1.tex", "Top-10 Stage-1 DCEE validation configs.")

        top_variants = []
        for _, row in stage1_summary.head(top_k).iterrows():
            variant_name = str(row["variant"])
            match = next(v for v in stage1_variants if v["variant"] == variant_name)
            top_variants.append(match)

        if args.skip_stage2 and (outdir / "dcee_tuning_stage2.csv").exists():
            stage2_summary = pd.read_csv(outdir / "dcee_tuning_stage2.csv")
        else:
            emit(f"Stage 2: top {top_k}, seeds={stage2_seeds}, slots={stage2_slots}", log)
            stage2_summary = run_variant_set(
                base,
                top_variants,
                stage2_seeds,
                stage2_slots,
                outdir,
                raw_name="dcee_tuning_stage2_raw.csv",
                slot_name="dcee_tuning_stage2_slot_metrics.csv",
                summary_name="dcee_tuning_stage2.csv",
                log_handle=log,
            )
        write_table(stage2_summary, tables_dir / "table_formal_dcee_tuning.tex", "Stage-2 DCEE validation configs.")

        best_name = str(stage2_summary.iloc[0]["variant"])
        best_variant = next(v for v in top_variants if v["variant"] == best_name)
        best_cfg = build_config(base, best_variant["overrides"])
        best_payload = {
            "selection_rule": "min Stage-2 validation mean penalty; tie-break lower backlog",
            "stage1_validation_seeds": stage1_seeds,
            "stage1_slots": stage1_slots,
            "stage2_validation_seeds": stage2_seeds,
            "stage2_slots": stage2_slots,
            "test_seeds": test_seeds,
            "test_slots": test_slots,
            "best_variant": best_name,
            "best_overrides": best_variant["overrides"],
            "scenario_config": scenario_config_to_dict(best_cfg),
        }
        (outdir / "best_dcee_config_formal.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        emit(f"Best formal DCEE config: {best_name}", log)

        if not args.skip_test:
            test_summaries = []
            test_slots_rows = []
            default_cfg = build_config(base, {})
            for label, cfg, variant in [
                ("DCEE-default", default_cfg, "default"),
                ("DCEE-tuned", best_cfg, best_name),
            ]:
                part_summaries, part_slots = evaluate_method(
                    cfg,
                    method="proposed_full",
                    label=label,
                    seeds=test_seeds,
                    slots=test_slots,
                    variant=variant,
                    log_handle=log,
                )
                test_summaries.extend(part_summaries)
                test_slots_rows.extend(part_slots)
            test_raw = pd.concat(test_summaries, ignore_index=True)
            test_raw.to_csv(outdir / "dcee_test_raw.csv", index=False)
            pd.concat(test_slots_rows, ignore_index=True).to_csv(outdir / "dcee_test_slot_metrics.csv", index=False)
            dcee_test_summary = summarize_seed_rows(test_raw, ["method_label", "method", "variant"])
            dcee_test_summary.to_csv(outdir / "dcee_test_summary.csv", index=False)
            emit(dcee_test_summary[["method_label", "total_penalty_mean", "queue_backlog_mean", "solver_success_mean"]].to_string(index=False), log)

        manifest = {
            "config": str(args.config),
            "stage1_configs": len(stage1_variants),
            "stage2_top_k": top_k,
            "best_variant": best_name,
            "elapsed_sec": time.perf_counter() - t0,
            "outputs": [
                str(outdir / "dcee_tuning_stage1.csv"),
                str(outdir / "dcee_tuning_stage2.csv"),
                str(outdir / "best_dcee_config_formal.json"),
                str(outdir / "dcee_test_summary.csv"),
                str(tables_dir / "table_formal_dcee_tuning.tex"),
            ],
        }
        (outdir / "dcee_tuning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(f"DCEE tuning elapsed={manifest['elapsed_sec']:.1f}s", log)


if __name__ == "__main__":
    main()
