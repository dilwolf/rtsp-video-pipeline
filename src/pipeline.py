"""RTSP -> YOLO detection -> ByteTrack -> annotated output, with stage timings.

Tracking is ultralytics' built-in ByteTrack. supervision's own ByteTrack is
deprecated as of 0.28 and the separate trackers package would be a third
dependency for an algorithm already sitting in the one we import anyway.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ffmpeg_reader import FFmpegReader
from rtsp_source import LatestFrame


def percentiles(values):
    if not values:
        return {}
    a = np.asarray(values) * 1e3
    return {"mean_ms": round(float(a.mean()), 2),
            "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "max_ms": round(float(a.max()), 2)}


def build_annotators(segment):
    ann = {"box": sv.BoxAnnotator(), "label": sv.LabelAnnotator(),
           "trace": sv.TraceAnnotator()}
    if segment:
        ann["mask"] = sv.MaskAnnotator()
    return ann


def annotate(frame, det, ann, names):
    out = frame
    if "mask" in ann and det.mask is not None:
        out = ann["mask"].annotate(out, det)
    out = ann["box"].annotate(out, det)
    if len(det):
        labels = [f"#{tid} {names.get(int(cid), cid)}"
                  for tid, cid in zip(
                      det.tracker_id if det.tracker_id is not None
                      else [-1] * len(det), det.class_id)]
        out = ann["label"].annotate(out, det, labels=labels)
        if det.tracker_id is not None:
            out = ann["trace"].annotate(out, det)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="rtsp://127.0.0.1:8554/cam1")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--segment", action="store_true",
                    help="use the -seg weights and draw masks")
    ap.add_argument("--device", default="0", help="'0' for cuda:0, or 'cpu'")
    ap.add_argument("--hwaccel", default=None,
                    help="ffmpeg decoder, e.g. cuda; omit for software decode")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=15,
                    help="frames excluded from the timings")
    ap.add_argument("--video-out", default=None, help="write an annotated mp4")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--json-out", default="outputs/latency.json")
    args = ap.parse_args()

    weights = args.model
    if args.segment and "-seg" not in weights:
        weights = weights.replace(".pt", "-seg.pt")
    model = YOLO(weights)
    names = model.names

    reader = FFmpegReader(args.url, hwaccel=args.hwaccel)
    print(f"{args.url}  {reader.width}x{reader.height} "
          f"{reader.info.get('codec')} @ {reader.info.get('fps'):.0f} fps  "
          f"hwaccel={args.hwaccel or 'none'}")
    print(f"model {weights} on device {args.device}, imgsz {args.imgsz}")

    writer = None
    if args.video_out:
        Path(args.video_out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 20.0, (reader.width, reader.height))

    ann = build_annotators(args.segment)
    infer_t, annotate_t, loop_t = [], [], []
    seen_ids, n = set(), 0
    src = LatestFrame(reader)
    t_start = time.perf_counter()

    try:
        while n < args.max_frames:
            frame = src.get()
            if frame is None:
                if src.ended:
                    break
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            frame = frame.copy()  # the buffer is a view over the pipe read
            result = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                 imgsz=args.imgsz, conf=args.conf,
                                 device=args.device, verbose=False)[0]
            t1 = time.perf_counter()

            det = sv.Detections.from_ultralytics(result)
            if det.tracker_id is not None:
                seen_ids.update(int(i) for i in det.tracker_id)
            out = annotate(frame, det, ann, names)
            t2 = time.perf_counter()

            n += 1
            if n > args.warmup:
                infer_t.append(t1 - t0)
                annotate_t.append(t2 - t1)
                loop_t.append(t2 - t0)

            if writer is not None:
                cv2.putText(out, f"{len(det)} obj  {len(seen_ids)} ids  "
                                 f"dropped {src.dropped}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(out)
            if args.show:
                cv2.imshow("pipeline", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        elapsed = time.perf_counter() - t_start
        decoded, dropped = src.decoded, src.dropped
        src.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if not infer_t:
        raise SystemExit("no frames measured; is the stream up?")

    stats = {
        "url": args.url, "model": weights, "device": args.device,
        "hwaccel": args.hwaccel or "none", "imgsz": args.imgsz,
        "resolution": [reader.width, reader.height],
        "frames_processed": n, "frames_decoded": decoded,
        "frames_dropped": dropped,
        "drop_rate_pct": round(100 * dropped / max(decoded, 1), 1),
        "unique_track_ids": len(seen_ids),
        "decode_fps": round(decoded / elapsed, 1),
        # what one frame costs, if a frame is always ready
        "capacity_fps": round(len(loop_t) / sum(loop_t), 1),
        # what actually came out end to end, waiting and contention included.
        # The gap between the two is the honest number.
        "processed_fps": round(n / elapsed, 1),
        "infer": percentiles(infer_t),
        "annotate": percentiles(annotate_t),
        "loop": percentiles(loop_t),
    }

    print(f"\n{'stage':<10} {'mean':>8} {'p50':>8} {'p95':>8} {'max':>8}")
    for stage in ("infer", "annotate", "loop"):
        s = stats[stage]
        print(f"{stage:<10} {s['mean_ms']:>8.2f} {s['p50_ms']:>8.2f} "
              f"{s['p95_ms']:>8.2f} {s['max_ms']:>8.2f}")
    print(f"\ndecode {stats['decode_fps']} fps   "
          f"capacity {stats['capacity_fps']} fps   "
          f"sustained {stats['processed_fps']} fps")
    print(f"decoded {decoded}, dropped {dropped} ({stats['drop_rate_pct']}%), "
          f"{len(seen_ids)} unique track ids")

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
