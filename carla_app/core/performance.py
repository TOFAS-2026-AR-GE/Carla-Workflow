"""Canlı döngü sürelerini düşük maliyetli hareketli ortalamayla izler."""


class PerformanceMonitor:
    """Kare bütçesi, görüntüleme ve algılama kuyruğu gecikmesini özetler."""

    def __init__(
        self,
        frame_budget_ms,
        smoothing=0.12,
        latency_budget_ms=80.0,
    ):
        self.frame_budget_ms = max(1.0, float(frame_budget_ms))
        self.latency_budget_ms = max(
            self.frame_budget_ms,
            float(latency_budget_ms),
        )
        self.smoothing = min(1.0, max(0.01, float(smoothing)))
        self.values = {}
        self.over_budget_frames = 0
        self.total_frames = 0
        self.over_latency_results = 0
        self.total_results = 0
        self.last_processed_count = -1

    def update(
        self,
        process_ms,
        viewer_ms,
        camera_wait_ms,
        perception_result,
        worker_diagnostics,
    ):
        self.total_frames += 1
        if float(process_ms) > self.frame_budget_ms:
            self.over_budget_frames += 1

        perception_result = perception_result or {}
        self._ema("process_ms", process_ms)
        self._ema("viewer_ms", viewer_ms)
        self._ema("camera_wait_ms", camera_wait_ms)
        processed_count = int(worker_diagnostics.get("processed", 0))
        if (
            perception_result
            and processed_count != self.last_processed_count
        ):
            inference_ms = float(perception_result.get("elapsed_ms", 0.0))
            queue_ms = float(perception_result.get("queue_delay_ms", 0.0))
            end_to_end_ms = float(
                perception_result.get(
                    "worker_total_ms",
                    inference_ms + queue_ms,
                )
            )
            self._ema("inference_ms", inference_ms)
            self._ema("queue_ms", queue_ms)
            self._ema("end_to_end_ms", end_to_end_ms)
            self.total_results += 1
            if end_to_end_ms > self.latency_budget_ms:
                self.over_latency_results += 1
            self.last_processed_count = processed_count

        self.values["dropped"] = int(worker_diagnostics.get("dropped", 0))
        self.values["processed"] = processed_count

    def summary(self):
        if self.total_frames == 0:
            return ""
        budget_ratio = 100.0 * self.over_budget_frames / self.total_frames
        latency_ratio = (
            100.0 * self.over_latency_results / self.total_results
            if self.total_results
            else 0.0
        )
        return (
            f" loop={self.values.get('process_ms', 0.0):.1f}ms"
            f" view={self.values.get('viewer_ms', 0.0):.1f}ms"
            f" infer={self.values.get('inference_ms', 0.0):.1f}ms"
            f" q={self.values.get('queue_ms', 0.0):.1f}ms"
            f" e2e={self.values.get('end_to_end_ms', 0.0):.1f}ms"
            f" drop={int(self.values.get('dropped', 0))}"
            f" budget_over={budget_ratio:.0f}%"
            f" latency_over={latency_ratio:.0f}%"
        )

    def _ema(self, name, value):
        value = max(0.0, float(value))
        previous = self.values.get(name)
        if previous is None:
            self.values[name] = value
            return
        self.values[name] = previous + self.smoothing * (value - previous)


class AdaptivePerceptionScheduler:
    """Algılama yetişemediğinde kare aralığını sınırlı biçimde ayarlar."""

    def __init__(
        self,
        frame_budget_ms,
        initial_period=1,
        maximum_period=3,
        evaluation_frames=40,
    ):
        self.frame_budget_ms = max(1.0, float(frame_budget_ms))
        self.period = max(1, int(initial_period))
        self.maximum_period = max(self.period, int(maximum_period))
        self.evaluation_frames = max(10, int(evaluation_frames))
        self.frames_since_evaluation = 0
        self.previous_submitted = 0
        self.previous_dropped = 0
        self.stable_windows = 0

    def update(self, inference_ms, worker_diagnostics):
        """Yeni öneri oluşursa periyodu, aksi halde ``None`` döndürür."""
        self.frames_since_evaluation += 1
        if self.frames_since_evaluation < self.evaluation_frames:
            return None
        self.frames_since_evaluation = 0

        submitted = int(worker_diagnostics.get("submitted", 0))
        dropped = int(worker_diagnostics.get("dropped", 0))
        submitted_delta = max(0, submitted - self.previous_submitted)
        dropped_delta = max(0, dropped - self.previous_dropped)
        self.previous_submitted = submitted
        self.previous_dropped = dropped

        drop_ratio = dropped_delta / max(1, submitted_delta)
        inference_ms = max(0.0, float(inference_ms))
        available_inference_ms = (
            self.frame_budget_ms * self.period * 0.90
        )
        overloaded = (
            drop_ratio >= 0.12
            or inference_ms > available_inference_ms
        )

        if overloaded and self.period < self.maximum_period:
            self.period += 1
            self.stable_windows = 0
            return self.period

        next_faster_period = max(1, self.period - 1)
        faster_budget_ms = (
            self.frame_budget_ms * next_faster_period * 0.65
        )
        stable = drop_ratio == 0.0 and inference_ms <= faster_budget_ms
        self.stable_windows = self.stable_windows + 1 if stable else 0
        if self.period > 1 and self.stable_windows >= 3:
            self.period -= 1
            self.stable_windows = 0
            return self.period
        return None
