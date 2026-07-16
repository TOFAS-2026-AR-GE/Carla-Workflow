"""Single-slot background worker for perception inference."""

import queue
import threading


class PerceptionWorker:
    """Always process the newest submitted frame and discard queued stale work."""

    def __init__(self, perception_system):
        self.system = perception_system
        self.queue = queue.Queue(maxsize=1)
        self.latest_result = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_error = None
        self.thread = threading.Thread(
            target=self._loop,
            name="perception-worker",
            daemon=True,
        )
        self.thread.start()

    def submit(self, frame_id, rgb_image):
        item = (int(frame_id), rgb_image)

        try:
            self.queue.put_nowait(item)
            return
        except queue.Full:
            pass

        # The queued frame is already stale. Replace it with the newest frame.
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(item)
        except queue.Full:
            pass

    def get_latest(self):
        with self.lock:
            return self.latest_result

    def stop(self):
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass

        self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            print("[WARN] Perception worker 3 saniyede durmadi.")

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                return

            frame_id, image = item
            try:
                result = self.system.detect(frame_id, image)
            except Exception as error:
                # PerceptionSystem isolates detector failures. Reaching this
                # block means the orchestration itself failed unexpectedly.
                message = f"{type(error).__name__}: {error}"
                if message != self.last_error:
                    print(f"[ERROR] Perception worker: {message}")
                self.last_error = message
                continue

            self.last_error = None
            with self.lock:
                self.latest_result = result
