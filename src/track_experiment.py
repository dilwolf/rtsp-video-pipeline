"""Log a pipeline run's measurements to MLflow.

Kept separate from pipeline.py so the pipeline has no reason to import mlflow
on a machine that only wants to run inference.
"""
import argparse
import json
from pathlib import Path

import mlflow


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latency", default="outputs/latency.json")
    ap.add_argument("--decode", default="outputs/decode_bench.json")
    ap.add_argument("--experiment", default="rtsp-pipeline")
    ap.add_argument("--tracking-uri", default=None)
    args = ap.parse_args()

    latency = load(args.latency)
    if latency is None:
        raise SystemExit(f"{args.latency} not found; run src/pipeline.py first")

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": latency["model"],
            "device": latency["device"],
            "hwaccel": latency["hwaccel"],
            "imgsz": latency["imgsz"],
            "resolution": "x".join(map(str, latency["resolution"])),
            "url": latency["url"],
        })
        mlflow.log_metrics({
            "infer_mean_ms": latency["infer"]["mean_ms"],
            "infer_p95_ms": latency["infer"]["p95_ms"],
            "annotate_mean_ms": latency["annotate"]["mean_ms"],
            "loop_p95_ms": latency["loop"]["p95_ms"],
            "decode_fps": latency["decode_fps"],
            "capacity_fps": latency["capacity_fps"],
            "processed_fps": latency["processed_fps"],
            "drop_rate_pct": latency["drop_rate_pct"],
            "unique_track_ids": latency["unique_track_ids"],
        })
        mlflow.log_artifact(args.latency)

        decode = load(args.decode)
        if decode:
            for row in decode["results"]:
                tag = row["hwaccel"]
                metrics = {f"decode_{tag}_fps": row["fps"],
                           f"decode_{tag}_ms": row["ms_per_frame"]}
                if row.get("decode_only"):
                    metrics[f"decode_{tag}_cpu_s"] = row["decode_only"]["cpu_s"]
                mlflow.log_metrics(metrics)
            mlflow.log_artifact(args.decode)

        print(f"logged run {run.info.run_id} to experiment '{args.experiment}'")
        print(f"  infer p95 {latency['infer']['p95_ms']} ms, "
              f"sustained {latency['processed_fps']} fps, "
              f"drops {latency['drop_rate_pct']}%")


if __name__ == "__main__":
    main()
