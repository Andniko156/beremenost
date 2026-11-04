import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

from .data import SCENARIOS, ExperimentConfig, read_csv, synthetic_data
from .evaluation import backtest
from .forecast import MODEL_ID, MODEL_REVISION, ChronosForecaster, PersistenceForecaster, benchmark_baselines, forecast_frame


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Сохраняя главное: эксперимент прогноза ЧСС")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Create synthetic engineering data")
    generate.add_argument("--output", default="data/synthetic_hr.csv")
    generate.add_argument("--days", type=int, default=3)
    generate.add_argument("--series", type=int, default=2)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--interval-minutes", type=int, default=5)
    generate.add_argument("--scenario", choices=SCENARIOS, default="daily")
    serve = sub.add_parser("serve", help="Run the private experimental forecast API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    prepare = sub.add_parser("prepare-hrv29", help="Prepare reviewed pregnancy HR data with subject-disjoint splits")
    prepare.add_argument("--input", help="Reviewed public Excel; downloaded if omitted")
    prepare.add_argument("--output-dir", default="data/hrv29")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--context-steps", type=int, default=6)
    prepare.add_argument("--prediction-steps", type=int, default=12)
    tune = sub.add_parser("finetune", help="Research-only LoRA adaptation; no API model promotion")
    tune.add_argument("--dataset", required=True)
    tune.add_argument("--output-dir", required=True)
    tune.add_argument("--init-model", default=MODEL_ID)
    tune.add_argument("--steps", type=int, default=100)
    tune.add_argument("--batch-size", type=int, default=4)
    tune.add_argument("--learning-rate", type=float, default=1e-5)
    tune.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    tune.add_argument("--cpu-threads", type=int, default=4)
    tune.add_argument("--seed", type=int, default=42)
    tune.add_argument("--offline", action="store_true")
    export = sub.add_parser("export-adapter", help="Restore a LoRA adapter as a merged model for another stage")
    export.add_argument("--input", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    export.add_argument("--offline", action="store_true")
    for command in ("forecast", "backtest", "benchmark"):
        run = sub.add_parser(command)
        run.add_argument("--input", required=True)
        run.add_argument("--output-dir", default=f"outputs/{command}")
        run.add_argument("--backend", choices=["chronos2", "persistence"], default="chronos2")
        run.add_argument("--interval-minutes", type=int, default=5)
        run.add_argument("--history-steps", type=int, default=288)
        run.add_argument("--horizon-steps", type=int, default=12)
        run.add_argument("--model", default=MODEL_ID)
        run.add_argument("--revision", default=MODEL_REVISION)
        run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        run.add_argument("--offline", action="store_true", help="Load cached/local weights without Hub requests")
        run.add_argument("--data-kind", choices=["synthetic", "device", "unknown"], default="unknown")
        if command in ("backtest", "benchmark"):
            run.add_argument("--folds", type=int, default=12 if command == "benchmark" else 3)
    return parser


def run(args):
    if args.command == "prepare-hrv29":
        from .research_data import ResearchProfile, prepare_hrv29
        manifest = prepare_hrv29(args.output_dir, args.input, args.seed,
                                 ResearchProfile(context_steps=args.context_steps,
                                                 prediction_steps=args.prediction_steps))
        print(json.dumps({"source": manifest["source"]["dataset_id"], "audit": manifest["audit"],
                          "subjects": {k: len(v) for k, v in manifest["splits"].items()}}, indent=2))
        return
    if args.command == "finetune":
        from .finetuning import finetune
        finetune(args.dataset, args.output_dir, init_model=args.init_model, num_steps=args.steps,
                 batch_size=args.batch_size, learning_rate=args.learning_rate, device=args.device,
                 cpu_threads=args.cpu_threads, seed=args.seed, offline=args.offline)
        return
    if args.command == "export-adapter":
        from .finetuning import export_adapter
        export_adapter(args.input, args.output_dir, args.device, args.offline)
        return
    if args.command == "serve":
        try:
            import uvicorn
            from .api import create_app
        except ImportError as exc:
            raise RuntimeError("Install API dependencies: python -m pip install -e '.[api]'") from exc
        uvicorn.run(create_app(), host=args.host, port=args.port, access_log=False)
        return
    if args.command == "generate":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = synthetic_data(args.days, args.series, args.seed, args.interval_minutes, args.scenario)
        frame.to_csv(path, index=False)
        write_json(path.with_suffix(".metadata.json"), {
            "data_kind": "synthetic", "seed": args.seed, "rows": len(frame),
            "interval_minutes": args.interval_minutes, "clinical_validity": "not evaluated",
            "scenario": args.scenario,
        })
        print(f"Synthetic data: {path} ({len(frame)} rows). Not patient observations.")
        return
    config = ExperimentConfig(args.interval_minutes, args.history_steps, args.horizon_steps)
    frame = read_csv(args.input, config)
    # Check evaluation length before spending time downloading/loading weights.
    baselines = benchmark_baselines(config) if args.command == "benchmark" else None
    if args.command in ("backtest", "benchmark"):
        required = config.history_steps + args.folds * config.horizon_steps
        if args.folds < 1 or (frame.groupby("series_id").size() < required).any():
            raise ValueError(f"Backtest requires positive folds and at least {required} rows per series")
    started = time.perf_counter()
    model = (PersistenceForecaster() if args.backend == "persistence" else
             ChronosForecaster(args.model, args.revision, args.device, args.offline))
    report = {}
    if args.command == "forecast":
        rows = forecast_frame(frame, config, model)
    else:
        rows, report = backtest(frame, config, model, args.folds, baselines=baselines)
    report.update({
        "experiment": asdict(config), "horizon_minutes": config.interval_minutes * config.horizon_steps,
        "model": model.metadata(), "data_kind": args.data_kind,
        "clinical_validity": "not evaluated", "elapsed_seconds_including_model_load": time.perf_counter() - started,
        "series_count": int(frame.series_id.nunique()),
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "runtime": {"python": sys.version.split()[0], "numpy": version("numpy"), "pandas": version("pandas")},
        "input_last_timestamps": {str(key): value.isoformat()
                                  for key, value in frame.groupby("series_id").timestamp.max().items()},
    })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "predictions.csv", index=False)
    write_json(output / "report.json", report)
    print(f"Backend: {model.name}; horizon: {report['horizon_minutes']} min; data: {args.data_kind}")
    for name, score in report.get("aggregate", {}).items():
        print(f"  {name}: MAE={score['mae_bpm']:.3f} bpm, RMSE={score['rmse_bpm']:.3f} bpm")
    print(f"Saved forecasts and full report to {output}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
