"""Decode throughput, software against hardware.

Deliberately run on a local file rather than the RTSP stream. The publisher
paces itself with -re, so decoding from it measures the publisher's 30 fps and
not the decoder.

Hardware decode does not make frames appear in a NumPy array for free. NVDEC
puts them in device memory and this pipeline wants them on the host, so every
frame pays a copy back across PCIe. Whether that trade wins depends on
resolution and codec, which is the point of measuring instead of assuming.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from ffmpeg_reader import FFmpegError, FFmpegReader, probe


def decode_cpu_cost(path, hwaccel, frames):
    """CPU seconds ffmpeg spends decoding, with no pipe in the way.

    Throughput alone makes hardware decode look pointless. It is not: on a box
    serving many cameras the scarce resource is CPU, and this is the number
    that says whether NVDEC bought any back.
    """
    cmd = ["ffmpeg", "-hide_banner", "-benchmark"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", path, "-frames:v", str(frames), "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True)

    found = {}
    for m in re.finditer(r"(utime|stime|rtime)=([\d.]+)s", out.stderr):
        found[m.group(1)] = float(m.group(2))
    if not found:
        return None
    cpu = found.get("utime", 0.0) + found.get("stime", 0.0)
    wall = found.get("rtime", 0.0)
    return {"cpu_s": round(cpu, 3), "wall_s": round(wall, 3),
            "cpu_ms_per_frame": round(1e3 * cpu / frames, 3),
            "decode_only_fps": round(frames / wall, 1) if wall else None}


def run(path, hwaccel, frames):
    reader = FFmpegReader(path, hwaccel=hwaccel)
    try:
        n, first = 0, None
        t0 = time.perf_counter()
        while n < frames:
            frame = reader.read()
            if frame is None:
                break
            if first is None:
                first = time.perf_counter() - t0  # includes decoder startup
            n += 1
        elapsed = time.perf_counter() - t0
    finally:
        tail = reader.stderr_tail
        reader.close()

    if n == 0:
        raise FFmpegError(f"decoded nothing with hwaccel={hwaccel}: {tail}")
    return {"hwaccel": hwaccel or "none", "frames": n,
            "elapsed_s": round(elapsed, 2),
            "fps": round(n / elapsed, 1),
            "ms_per_frame": round(1e3 * elapsed / n, 3),
            "startup_s": round(first, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/sample_1080p.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--accels", nargs="+", default=["none", "cuda", "d3d11va"])
    ap.add_argument("--json-out", default="outputs/decode_bench.json")
    args = ap.parse_args()

    info = probe(args.input)
    print(f"{args.input}  {info['width']}x{info['height']}  {info['codec']}  "
          f"{args.frames} frames\n")

    rows = []
    for name in args.accels:
        hwaccel = None if name == "none" else name
        try:
            row = run(args.input, hwaccel, args.frames)
        except FFmpegError as e:
            print(f"{name:<10} unavailable: {str(e).splitlines()[-1][:60]}")
            continue
        row["decode_only"] = decode_cpu_cost(args.input, hwaccel, args.frames)
        rows.append(row)
        cpu = row["decode_only"]
        cpu_txt = (f"  decode-only {cpu['decode_only_fps']:>6.1f} fps, "
                   f"CPU {cpu['cpu_s']:>5.2f}s" if cpu else "")
        print(f"{row['hwaccel']:<10} {row['fps']:>8.1f} fps through the pipe  "
              f"{row['ms_per_frame']:>7.3f} ms/frame{cpu_txt}")

    if rows:
        base = next((r for r in rows if r["hwaccel"] == "none"), rows[0])
        bcpu = base.get("decode_only")
        print(f"\nrelative to {base['hwaccel']}")
        for r in rows:
            cpu = r.get("decode_only")
            saved = (f"{bcpu['cpu_s'] / cpu['cpu_s']:.1f}x less decode CPU"
                     if cpu and bcpu and cpu["cpu_s"] else "-")
            print(f"  {r['hwaccel']:<10} {r['fps'] / base['fps']:.2f}x throughput"
                  f"   {saved}")

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"input": args.input, "resolution": [info["width"], info["height"]],
         "codec": info["codec"], "results": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
