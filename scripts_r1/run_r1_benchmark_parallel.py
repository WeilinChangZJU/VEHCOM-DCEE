from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


def parse_seed_values(text: str | None, config: Dict[str, object]) -> List[int]:
    if text:
        values: List[int] = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if ".." in part:
                start, end = [int(x) for x in part.split("..", 1)]
                values.extend(range(start, end + 1))
            else:
                values.append(int(part))
        return values
    configured = config.get("seed_values")
    if configured is not None:
        return [int(x) for x in configured]
    seeds = int(config.get("seeds", 1))
    return list(range(seeds))


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_manifest(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def seed_dir_name(seed_idx: int) -> str:
    return f"seed_{seed_idx:03d}"


def run_seed(
    *,
    python_exe: str,
    config_path: Path,
    outdir: Path,
    seed_idx: int,
    slots: int,
    log_dir: Path,
    resume: bool,
    rerun_failed: bool,
    rerun_paused: bool,
) -> Dict[str, object]:
    seed_outdir = outdir / seed_dir_name(seed_idx)
    manifest_path = seed_outdir / "run_manifest.json"
    existing = read_manifest(manifest_path)
    existing_status = str(existing.get("status", "")) if existing else ""
    if resume and existing and existing.get("status") == "completed":
        return {"seed": seed_idx, "status": "skipped_completed", "outdir": str(seed_outdir)}
    if resume and existing and existing.get("status") == "failed" and not rerun_failed:
        return {"seed": seed_idx, "status": "skipped_failed", "outdir": str(seed_outdir)}
    if resume and existing and existing.get("status") == "paused" and not rerun_paused:
        return {"seed": seed_idx, "status": "skipped_paused", "outdir": str(seed_outdir)}
    if existing and existing_status != "completed":
        shutil.rmtree(seed_outdir, ignore_errors=True)

    seed_outdir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"patch3a_seed_{seed_idx:03d}.log"
    started = time.time()
    write_manifest(manifest_path, {
        "seed": seed_idx,
        "seed_value": 1000 + seed_idx,
        "slots": slots,
        "status": "running",
        "started_at": started,
    })

    cmd = [
        python_exe,
        str(ROOT / "scripts_r1" / "run_r1_benchmark_repair.py"),
        "--config",
        str(config_path),
        "--slots",
        str(slots),
        "--seeds",
        "1",
        "--seed-offset",
        str(1000 + seed_idx),
        "--outdir",
        str(seed_outdir),
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("Command: " + " ".join(cmd) + "\n")
        log_handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    finished = time.time()
    status = "completed" if proc.returncode == 0 else "failed"
    manifest = {
        "seed": seed_idx,
        "seed_value": 1000 + seed_idx,
        "slots": slots,
        "status": status,
        "returncode": proc.returncode,
        "started_at": started,
        "finished_at": finished,
        "elapsed_seconds": finished - started,
        "outdir": str(seed_outdir),
        "log": str(log_path),
    }
    write_manifest(manifest_path, manifest)
    if proc.returncode != 0:
        raise RuntimeError(f"Seed {seed_idx} failed with exit code {proc.returncode}. See {log_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FC-Union benchmark jobs in parallel by seed.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=str(ROOT / "results_r1" / "benchmark_full_balanced"))
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma list or ranges, e.g. 0..19 or 0,1,2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--rerun-paused", action="store_true")
    parser.add_argument("--log-dir", type=str, default=str(ROOT / "logs_r1"))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    slots = int(args.slots if args.slots is not None else config.get("slots", 200))
    seeds = parse_seed_values(args.seeds, config)
    outdir = Path(args.outdir)
    log_dir = Path(args.log_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(f"Parallel FC-Union start: seeds={seeds}, slots={slots}, workers={args.workers}, outdir={outdir}")
    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [
            pool.submit(
                run_seed,
                python_exe=args.python,
                config_path=config_path,
                outdir=outdir,
                seed_idx=seed_idx,
                slots=slots,
                log_dir=log_dir,
                resume=args.resume,
                rerun_failed=args.rerun_failed,
                rerun_paused=args.rerun_paused,
            )
            for seed_idx in seeds
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"seed={result['seed']} status={result['status']} elapsed={result.get('elapsed_seconds', 0):.1f}s")

    elapsed = time.time() - started
    summary_path = outdir / "parallel_run_summary.json"
    summary_path.write_text(json.dumps({
        "seeds": seeds,
        "slots": slots,
        "workers": args.workers,
        "elapsed_seconds": elapsed,
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Parallel FC-Union finished in {elapsed:.1f}s")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
