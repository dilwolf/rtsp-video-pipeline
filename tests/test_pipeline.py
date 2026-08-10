"""Hardware-free tests: no camera, no GPU, no RTSP server.

Anything needing a real decoder is skipped when ffmpeg is absent, so the suite
still means something in CI.
"""
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffmpeg_reader import FFmpegError, FFmpegReader, probe
from rtsp_source import LatestFrame

has_ffmpeg = shutil.which("ffmpeg") and shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg not on PATH")


class FakeReader:
    """Produces numbered frames on demand, then ends."""

    def __init__(self, count, delay=0.0):
        self.count = count
        self.delay = delay
        self.i = 0
        self.closed = False

    def read(self):
        if self.i >= self.count:
            return None
        if self.delay:
            time.sleep(self.delay)
        self.i += 1
        return np.full((2, 2, 3), self.i % 256, np.uint8)

    def close(self):
        self.closed = True


def test_latest_frame_serves_newest_and_counts_drops():
    src = LatestFrame(FakeReader(20))
    while not src.ended:
        time.sleep(0.005)

    frame = src.get()
    assert frame is not None
    assert int(frame[0, 0, 0]) == 20 % 256, "should serve the newest, not the oldest"
    assert src.decoded == 20
    assert src.dropped == 19, "every unconsumed frame counts once"
    src.close()


def test_get_returns_none_once_consumed():
    src = LatestFrame(FakeReader(1))
    while not src.ended:
        time.sleep(0.005)
    assert src.get() is not None
    assert src.get() is None, "a consumed frame is not served twice"
    src.close()


def test_slow_consumer_drops_instead_of_lagging():
    """The point of the whole class: when inference is slower than the stream,
    frames are discarded so results stay current."""
    src = LatestFrame(FakeReader(40, delay=0.002))
    seen = []
    for _ in range(4):
        time.sleep(0.02)  # far slower than the producer
        f = src.get()
        if f is not None:
            seen.append(int(f[0, 0, 0]))
    src.close()

    assert len(seen) >= 2
    assert seen == sorted(seen), "frames arrive in order, never backwards"
    assert src.dropped > 0, "a slow consumer must be dropping"


def test_close_is_idempotent_and_closes_reader():
    reader = FakeReader(5)
    src = LatestFrame(reader)
    src.close()
    src.close()
    assert reader.closed


def test_no_thread_left_running_after_close():
    before = threading.active_count()
    LatestFrame(FakeReader(3)).close()
    time.sleep(0.05)
    assert threading.active_count() <= before


@pytest.fixture(scope="module")
def tiny_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=160x120:rate=10:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(path)], check=True)
    return str(path)


@needs_ffmpeg
def test_probe_reports_stream_geometry(tiny_clip):
    info = probe(tiny_clip)
    assert (info["width"], info["height"]) == (160, 120)
    assert info["codec"] == "h264"
    assert info["fps"] == pytest.approx(10.0, abs=0.1)


@needs_ffmpeg
def test_probe_raises_on_missing_input():
    with pytest.raises(FFmpegError):
        probe("does_not_exist.mp4")


@needs_ffmpeg
def test_reader_yields_correctly_shaped_frames(tiny_clip):
    with FFmpegReader(tiny_clip) as reader:
        assert (reader.width, reader.height) == (160, 120)
        frame = reader.read()
        assert frame.shape == (120, 160, 3)
        assert frame.dtype == np.uint8


@needs_ffmpeg
def test_reader_frames_are_independent(tiny_clip):
    """Two reads must not alias one buffer; LatestFrame holds frames across
    reads and would otherwise hand out something already overwritten."""
    with FFmpegReader(tiny_clip) as reader:
        a = reader.read().copy()
        first = reader.read()
        second = reader.read()
        assert first is not second
        assert not np.shares_memory(first, second)
        assert a.shape == first.shape


@needs_ffmpeg
def test_reader_returns_none_at_end_of_stream(tiny_clip):
    with FFmpegReader(tiny_clip) as reader:
        count = 0
        while reader.read() is not None:
            count += 1
            if count > 200:
                pytest.fail("stream never ended")
    assert count == pytest.approx(10, abs=2)
