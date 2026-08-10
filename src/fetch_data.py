"""Fetch the sample clip and build the 1080p H.264 file the decode bench uses."""
import argparse
import subprocess
import urllib.request
from pathlib import Path

SAMPLE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    args = ap.parse_args()

    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)

    clip = data / "vtest.avi"
    if not clip.exists():
        print(f"fetching {SAMPLE}")
        urllib.request.urlretrieve(SAMPLE, clip)
    print(f"{clip} ready")

    # vtest is msmpeg4v3 at 768x576. NVDEC does not accelerate that codec at
    # all, so the decode benchmark needs real H.264 at a resolution where
    # decoding actually costs something.
    big = data / "sample_1080p.mp4"
    if not big.exists():
        print("building 1080p H.264 sample")
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-stream_loop", "3",
             "-i", str(clip), "-vf", "scale=1920:1080:flags=bicubic",
             "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-g", "60",
             "-pix_fmt", "yuv420p", "-an", "-y", str(big)], check=True)
    print(f"{big} ready ({big.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
