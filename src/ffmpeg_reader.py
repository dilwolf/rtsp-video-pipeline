"""Decode a video or RTSP stream by piping raw frames out of ffmpeg."""
import json
import shutil
import subprocess
import threading

import numpy as np


class FFmpegError(RuntimeError):
    pass


def probe(url, transport="tcp", timeout=15):
    """Frame size and rate, straight from the stream.

    Worth the extra process: hand the wrong dimensions to a rawvideo pipe and
    every frame is reshaped at the wrong stride, which shows up as a sheared
    image rather than an error.
    """
    if not shutil.which("ffprobe"):
        raise FFmpegError("ffprobe not found on PATH")
    cmd = ["ffprobe", "-v", "error"]
    if url.startswith("rtsp://"):
        cmd += ["-rtsp_transport", transport]
    cmd += ["-select_streams", "v:0", "-show_entries",
            "stream=width,height,avg_frame_rate,codec_name",
            "-of", "json", url]

    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {url}: {out.stderr.strip()}")

    streams = json.loads(out.stdout).get("streams") or []
    if not streams:
        raise FFmpegError(f"no video stream in {url}")
    s = streams[0]

    num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": fps, "codec": s.get("codec_name", "?")}


class FFmpegReader:
    """Frames as BGR NumPy arrays.

    bgr24 out of ffmpeg is already the byte order OpenCV expects, so the frame
    is a reshape of the pipe buffer with no conversion.
    """

    def __init__(self, url, hwaccel=None, transport="tcp", size=None, loglevel="error"):
        info = {"width": size[0], "height": size[1]} if size else probe(url, transport)
        self.width, self.height = info["width"], info["height"]
        self.info = info
        self.frame_bytes = self.width * self.height * 3
        self.url = url
        self.hwaccel = hwaccel

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", loglevel]
        if hwaccel:
            # decode on the GPU; ffmpeg still copies back to system memory
            # because the output format below lives on the host
            cmd += ["-hwaccel", hwaccel]
        if url.startswith("rtsp://"):
            # TCP: RTP over UDP drops packets under load and the artefacts land
            # in the decoded frame, where a detector happily reports on them
            cmd += ["-rtsp_transport", transport,
                    "-fflags", "nobuffer", "-flags", "low_delay"]
        cmd += ["-i", url, "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "pipe:1"]

        # bufsize=0. A large bufsize here is a trap: at 10**8 this pipe runs at
        # 10 fps against 160 fps unbuffered on the same 1080p file, because
        # every read drags Python's BufferedReader through a 100 MB buffer.
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        # drain stderr on a thread; a full pipe would otherwise deadlock ffmpeg
        self._err = []
        self._err_thread = threading.Thread(target=self._drain, daemon=True)
        self._err_thread.start()

    def _drain(self):
        for line in self.proc.stderr:
            self._err.append(line.decode(errors="replace").rstrip())
            del self._err[:-40]

    def read(self):
        """Next frame as HxWx3 BGR, or None once the stream ends.

        Reads straight into a fresh array rather than into one reused buffer.
        A shared buffer would be marginally faster and would hand every
        consumer a view that the next read overwrites underneath them --
        LatestFrame in particular holds frames across reads.
        """
        frame = np.empty((self.height, self.width, 3), np.uint8)
        view = memoryview(frame.reshape(-1))
        got = 0
        while got < self.frame_bytes:
            # an unbuffered pipe returns what is available, not what was asked
            n = self.proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n
        return frame

    @property
    def stderr_tail(self):
        return "\n".join(self._err)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for pipe in (self.proc.stdout, self.proc.stderr):
            if pipe and not pipe.closed:
                pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
