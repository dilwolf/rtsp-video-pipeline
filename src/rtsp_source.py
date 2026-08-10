"""Keep only the newest frame, so a slow consumer falls behind in time rather
than in latency."""
import threading


class LatestFrame:
    """Decode on a background thread and hand out the most recent frame.

    A camera does not wait. If inference is slower than the stream, something
    has to give: either frames queue up and every result describes an older and
    older moment, or old frames are dropped and results stay current. For live
    video the second is almost always what is wanted, so this keeps a single
    slot and counts what it overwrites.
    """

    def __init__(self, reader):
        self.reader = reader
        self._lock = threading.Lock()
        self._frame = None
        self.dropped = 0
        self.decoded = 0
        self.stopped = False
        self.ended = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self.stopped:
            frame = self.reader.read()
            if frame is None:
                self.ended = True
                break
            with self._lock:
                if self._frame is not None:
                    self.dropped += 1  # the consumer never saw the previous one
                self._frame = frame
                self.decoded += 1

    def get(self):
        """Newest frame, or None if the consumer has caught up."""
        with self._lock:
            frame, self._frame = self._frame, None
            return frame

    def close(self):
        self.stopped = True
        self.reader.close()
        self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
