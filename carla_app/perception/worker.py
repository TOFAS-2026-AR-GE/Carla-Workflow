"""Algılamayı ana araç döngüsünü bekletmeden arka planda çalıştırır."""

import queue
import threading
import time


class PerceptionWorker:
    """Kuyrukta yalnızca en yeni kamera karesini tutar."""

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
        item = {
            "kind": "single",
            "frame_id": int(frame_id),
            "image": rgb_image,
            "submitted_at": time.perf_counter(),
        }
        self.submit_item(item)

    def submit_cameras(self, camera_packet, primary_camera_name):
        """BEV modundaki kamera paketini en yeni iş olarak kuyruğa koyar."""
        item = {
            "kind": "cameras",
            "camera_packet": camera_packet,
            "primary_camera_name": primary_camera_name,
            "submitted_at": time.perf_counter(),
        }
        self.submit_item(item)

    def submit_item(self, item):
        """Eski bekleyen işi atıp yalnızca en yeni algılama işini tutar."""

        try:
            self.queue.put_nowait(item)
            return
        except queue.Full:
            pass

        # Kuyruktaki görüntü artık eskidir; yerine en yeni görüntüyü koy.
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

            try:
                started_at = time.perf_counter()
                submitted_at = float(item.get("submitted_at", started_at))
                if item["kind"] == "cameras":
                    result = self.system.detect_cameras(
                        item["camera_packet"],
                        item["primary_camera_name"],
                    )
                else:
                    result = self.system.detect(
                        item["frame_id"],
                        item["image"],
                    )
                result["queue_delay_ms"] = (
                    started_at - submitted_at
                ) * 1000.0
                result["worker_total_ms"] = (
                    time.perf_counter() - submitted_at
                ) * 1000.0
            except Exception as error:
                # Model hataları PerceptionSystem içinde ele alınır. Buraya
                # gelinmesi algılama akışının beklenmedik biçimde bozulduğunu
                # gösterir.
                message = f"{type(error).__name__}: {error}"
                if message != self.last_error:
                    print(f"[ERROR] Perception worker: {message}")
                self.last_error = message
                continue

            self.last_error = None
            with self.lock:
                self.latest_result = result
