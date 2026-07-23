"""CARLA için klasik UFLD şerit algılama adaptörü.

Kaynak model: https://huggingface.co/jkdxbns/autonomous-driving-carla
Özgün decode: https://github.com/cfzd/Ultra-Fast-Lane-Detection
"""

import time
from pathlib import Path

import cv2
import numpy as np

from carla_app.perception.device import (
    is_cuda_device,
    recover_cuda_memory,
    resolve_device,
)

MODEL_WIDTH = 800
MODEL_HEIGHT = 288
GRID_SIZE = 100
ROW_COUNT = 56
LANE_COUNT = 4
ROW_ANCHORS = np.linspace(115, 287, ROW_COUNT).astype(np.int32)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
CURVE_SAMPLE_STEP_PX = 4
CURVE_MINIMUM_RESIDUAL_PX = 4.0


def prepare_ufld_input(rgb_image):
    """RGB uint8 görüntüyü UFLD'nin NCHW ImageNet girdisine dönüştürür."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("UFLD girdisi HxWx3 RGB goruntu olmali.")

    resized = cv2.resize(
        image[:, :, :3],
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...])


def _softmax(values, axis=0):
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=axis, keepdims=True)


def decode_ufld_logits(
    logits,
    image_width,
    image_height,
    minimum_points=3,
    minimum_confidence=0.30,
):
    """UFLD 101x56x4 logitslerini kaynak görüntü koordinatlarına çözer."""
    output = np.asarray(logits, dtype=np.float32)
    if output.ndim == 4:
        if output.shape[0] != 1:
            raise ValueError("UFLD decode yalniz tek goruntuluk batch kabul eder.")
        output = output[0]

    expected_shape = (GRID_SIZE + 1, ROW_COUNT, LANE_COUNT)
    if output.shape != expected_shape:
        raise ValueError(
            f"UFLD cikti sekli {expected_shape} olmali; gelen: {output.shape}"
        )

    valid_probabilities = _softmax(output[:-1], axis=0)
    all_probabilities = _softmax(output, axis=0)
    grid_indices = np.arange(1, GRID_SIZE + 1, dtype=np.float32)[:, None, None]
    locations = np.sum(valid_probabilities * grid_indices, axis=0)
    selected_classes = np.argmax(output, axis=0)
    locations[selected_classes == GRID_SIZE] = 0.0

    column_step = (MODEL_WIDTH - 1.0) / (GRID_SIZE - 1.0)
    lanes = []
    for lane_index in range(LANE_COUNT):
        points = []
        point_confidences = []
        for row_index, anchor in enumerate(ROW_ANCHORS):
            location = float(locations[row_index, lane_index])
            if location <= 0.0:
                continue

            class_index = int(selected_classes[row_index, lane_index])
            confidence = float(
                all_probabilities[class_index, row_index, lane_index]
            )
            model_x = location * column_step - 1.0
            x = int(round(model_x * float(image_width) / MODEL_WIDTH))
            y = int(round(float(anchor) * float(image_height) / MODEL_HEIGHT)) - 1
            points.append(
                [
                    int(np.clip(x, 0, max(0, int(image_width) - 1))),
                    int(np.clip(y, 0, max(0, int(image_height) - 1))),
                ]
            )
            point_confidences.append(confidence)

        lane_confidence = (
            float(np.mean(point_confidences)) if point_confidences else 0.0
        )
        detected = (
            len(points) >= int(minimum_points)
            and lane_confidence >= float(minimum_confidence)
        )
        lanes.append(
            {
                "lane_index": lane_index,
                "points": points,
                "point_confidences": point_confidences,
                "confidence": lane_confidence,
                "detected": detected,
            }
        )

    return lanes


def fit_lane_curve(
    points,
    point_confidences,
    image_width,
    image_height,
    sample_step_px=CURVE_SAMPLE_STEP_PX,
):
    """Seyrek UFLD noktalarına güven ağırlıklı dayanıklı x(y) eğrisi uydurur.

    UFLD satır ankrajları doğal olarak seyrektir. Ham noktaları doğrudan
    birleştirmek tek bir yanlış ızgara seçiminin ekranda keskin bir kırık
    oluşturmasına yol açar. Burada yalnız gözlenen y aralığında ikinci derece
    eğri uydurulur; medyan mutlak sapma ile aykırı noktalar iki kez elenir.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    confidences = np.asarray(point_confidences, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        return None
    if len(coordinates) < 3 or len(confidences) != len(coordinates):
        return None

    valid = (
        np.all(np.isfinite(coordinates), axis=1)
        & np.isfinite(confidences)
        & (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] < float(image_width))
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] < float(image_height))
    )
    coordinates = coordinates[valid]
    confidences = confidences[valid]
    if len(coordinates) < 3:
        return None

    order = np.argsort(coordinates[:, 1])
    coordinates = coordinates[order]
    confidences = np.clip(confidences[order], 0.01, 1.0)
    unique_y = np.unique(coordinates[:, 1])
    if len(unique_y) < 3:
        return None

    merged_points = []
    merged_confidences = []
    for y_value in unique_y:
        selection = coordinates[:, 1] == y_value
        weights = confidences[selection]
        merged_points.append(
            [
                float(np.average(coordinates[selection, 0], weights=weights)),
                float(y_value),
            ]
        )
        merged_confidences.append(float(np.max(weights)))

    coordinates = np.asarray(merged_points, dtype=np.float64)
    confidences = np.asarray(merged_confidences, dtype=np.float64)
    center_y = float(np.mean(coordinates[:, 1]))
    scale_y = max(1.0, float(np.ptp(coordinates[:, 1])) / 2.0)
    normalized_y = (coordinates[:, 1] - center_y) / scale_y
    inliers = np.ones(len(coordinates), dtype=bool)
    coefficients = None

    for _iteration in range(3):
        if np.count_nonzero(inliers) < 3:
            break
        degree = min(2, np.count_nonzero(inliers) - 1)
        coefficients = np.polyfit(
            normalized_y[inliers],
            coordinates[inliers, 0],
            degree,
            w=np.sqrt(confidences[inliers]),
        )
        fitted_x = np.polyval(coefficients, normalized_y)
        residuals = np.abs(coordinates[:, 0] - fitted_x)
        inlier_residuals = residuals[inliers]
        median = float(np.median(inlier_residuals))
        mad = float(np.median(np.abs(inlier_residuals - median)))
        robust_sigma = 1.4826 * mad
        residual_limit = max(
            CURVE_MINIMUM_RESIDUAL_PX * float(image_width) / MODEL_WIDTH,
            median + 2.5 * robust_sigma,
        )
        updated = residuals <= residual_limit
        if np.count_nonzero(updated) < 3 or np.array_equal(updated, inliers):
            break
        inliers = updated

    if coefficients is None or np.count_nonzero(inliers) < 3:
        return None

    degree = min(2, np.count_nonzero(inliers) - 1)
    coefficients = np.polyfit(
        normalized_y[inliers],
        coordinates[inliers, 0],
        degree,
        w=np.sqrt(confidences[inliers]),
    )
    inlier_prediction = np.polyval(coefficients, normalized_y[inliers])
    weighted_squared_error = np.average(
        np.square(coordinates[inliers, 0] - inlier_prediction),
        weights=confidences[inliers],
    )

    y_min = int(round(float(np.min(coordinates[inliers, 1]))))
    y_max = int(round(float(np.max(coordinates[inliers, 1]))))
    step = max(2, int(sample_step_px))
    sample_y = np.arange(y_min, y_max + 1, step, dtype=np.float64)
    if len(sample_y) == 0 or sample_y[-1] != y_max:
        sample_y = np.append(sample_y, float(y_max))
    sample_x = np.polyval(coefficients, (sample_y - center_y) / scale_y)
    sample_x = np.clip(sample_x, 0.0, max(0.0, float(image_width) - 1.0))
    dense_points = np.column_stack((sample_x, sample_y))

    return {
        "points": np.rint(dense_points).astype(np.int32).tolist(),
        "inlier_count": int(np.count_nonzero(inliers)),
        "rejected_count": int(len(coordinates) - np.count_nonzero(inliers)),
        "fit_rms_px": float(np.sqrt(weighted_squared_error)),
    }


class LaneCurveTracker:
    """Görsel şerit eğrilerini inference kareleri arasında yumuşatır."""

    def __init__(self, smoothing=0.65, maximum_missing_frames=1):
        self.smoothing = min(1.0, max(0.0, float(smoothing)))
        self.maximum_missing_frames = max(0, int(maximum_missing_frames))
        self.tracks = {}
        self.image_size = None

    def update(self, lanes, image_width, image_height):
        """Ham lane sözlüklerini çizime hazır, yoğun ve kararlı hale getirir."""
        image_size = (int(image_width), int(image_height))
        if self.image_size != image_size:
            self.tracks.clear()
            self.image_size = image_size

        processed = []
        seen_indices = set()
        for lane in lanes:
            lane_index = int(lane.get("lane_index", len(processed)))
            seen_indices.add(lane_index)
            current = dict(lane)
            current["raw_points"] = [
                list(point) for point in lane.get("points", [])
            ]
            fitted = None
            if lane.get("detected"):
                fitted = fit_lane_curve(
                    lane.get("points", []),
                    lane.get("point_confidences", []),
                    image_width,
                    image_height,
                )

            if fitted is None:
                held = self._held_lane(lane_index, current)
                processed.append(held)
                continue

            dense_points = self._smooth_points(
                lane_index,
                fitted["points"],
            )
            current.update(fitted)
            current["points"] = dense_points
            current["detected"] = len(dense_points) >= 2
            current["temporally_held"] = False
            self.tracks[lane_index] = {
                "points": [list(point) for point in dense_points],
                "confidence": float(current.get("confidence", 0.0)),
                "missing_frames": 0,
                "fit_rms_px": float(current.get("fit_rms_px", 0.0)),
            }
            processed.append(current)

        for lane_index in list(self.tracks):
            if lane_index not in seen_indices:
                self.tracks[lane_index]["missing_frames"] += 1
                if (
                    self.tracks[lane_index]["missing_frames"]
                    > self.maximum_missing_frames
                ):
                    self.tracks.pop(lane_index, None)
        return processed

    def _smooth_points(self, lane_index, current_points):
        previous = self.tracks.get(lane_index)
        if previous is None:
            return [list(point) for point in current_points]

        old = np.asarray(previous["points"], dtype=np.float64)
        current = np.asarray(current_points, dtype=np.float64)
        if len(old) < 2 or len(current) < 2:
            return [list(point) for point in current_points]
        old_order = np.argsort(old[:, 1])
        old = old[old_order]
        previous_x = np.interp(
            current[:, 1],
            old[:, 1],
            old[:, 0],
            left=np.nan,
            right=np.nan,
        )
        overlap = np.isfinite(previous_x)
        current[overlap, 0] = (
            self.smoothing * current[overlap, 0]
            + (1.0 - self.smoothing) * previous_x[overlap]
        )
        return np.rint(current).astype(np.int32).tolist()

    def _held_lane(self, lane_index, current):
        previous = self.tracks.get(lane_index)
        if previous is None:
            current["detected"] = False
            current["points"] = []
            current["temporally_held"] = False
            return current

        previous["missing_frames"] += 1
        if previous["missing_frames"] > self.maximum_missing_frames:
            self.tracks.pop(lane_index, None)
            current["detected"] = False
            current["points"] = []
            current["temporally_held"] = False
            return current

        current["points"] = [list(point) for point in previous["points"]]
        current["confidence"] = float(previous["confidence"]) * 0.75
        current["fit_rms_px"] = float(previous["fit_rms_px"])
        current["detected"] = True
        current["temporally_held"] = True
        current["inlier_count"] = 0
        current["rejected_count"] = 0
        return current


class LaneDetector:
    """Ön RGB kamerada UFLD çıkarımı yapar; kontrol komutu üretmez."""

    def __init__(
        self,
        model_path,
        device="auto",
        minimum_points=3,
        minimum_confidence=0.30,
        use_half=True,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"UFLD model dosyasi bulunamadi: {path}")

        import torch

        from carla_app.perception.ufld_model import build_ufld_resnet18

        requested_device = resolve_device(
            device,
            minimum_free_memory_mb=1100.0,
        )
        if str(requested_device).isdigit():
            requested_device = f"cuda:{requested_device}"
        self.device = str(requested_device)
        self.cuda_enabled = is_cuda_device(self.device)
        self.use_half = bool(use_half and self.cuda_enabled)
        self.minimum_points = max(3, int(minimum_points))
        self.minimum_confidence = float(minimum_confidence)
        self.torch = torch
        self.curve_tracker = LaneCurveTracker()

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = self._state_dict_from_checkpoint(checkpoint)
        model = build_ufld_resnet18(
            grid_size=GRID_SIZE,
            row_count=ROW_COUNT,
            lane_count=LANE_COUNT,
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            names = ", ".join(incompatible.missing_keys[:5])
            raise RuntimeError(f"UFLD checkpoint katmanlari eksik: {names}")
        unexpected = [
            name
            for name in incompatible.unexpected_keys
            if not name.startswith(("aux_header", "aux_combine"))
        ]
        if unexpected:
            names = ", ".join(unexpected[:5])
            raise RuntimeError(f"UFLD checkpoint uyumsuz katmanlari: {names}")

        self.model = model.eval()
        self.host_buffer = None
        self.host_buffer_shape = None
        if self.cuda_enabled:
            try:
                self._activate_cuda()
            except Exception as error:
                if not self._is_cuda_memory_error(error):
                    raise
                self._fallback_to_cpu(error)
        else:
            self.model = self.model.to("cpu").float()
        print(
            f"[OK] UFLD modeli: device={self.device}, "
            f"fp16={self.use_half}"
        )

    @staticmethod
    def _state_dict_from_checkpoint(checkpoint):
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "model", "state_dict"):
                candidate = checkpoint.get(key)
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break
        if not isinstance(state_dict, dict):
            raise RuntimeError("UFLD checkpoint bir durum sozlugu icermiyor.")
        return {
            name[7:] if name.startswith("module.") else name: value
            for name, value in state_dict.items()
        }

    def detect(self, rgb_image):
        started_at = time.perf_counter()
        tensor = self._prepare_tensor(rgb_image)
        try:
            with self.torch.inference_mode():
                logits = self.model(tensor)
        except Exception as error:
            if not self.cuda_enabled or not self._is_cuda_memory_error(error):
                raise
            self._fallback_to_cpu(error)
            tensor = self._prepare_tensor(rgb_image)
            with self.torch.inference_mode():
                logits = self.model(tensor)
        logits = logits.detach().float().cpu().numpy()
        height, width = rgb_image.shape[:2]
        raw_lanes = decode_ufld_logits(
            logits,
            image_width=width,
            image_height=height,
            minimum_points=self.minimum_points,
            minimum_confidence=self.minimum_confidence,
        )
        lanes = self.curve_tracker.update(raw_lanes, width, height)
        return {
            "available": True,
            "model": "ufld_resnet18_carla",
            "model_input_size": [MODEL_WIDTH, MODEL_HEIGHT],
            "image_size": [int(width), int(height)],
            "lanes": lanes,
            "detected_count": sum(lane["detected"] for lane in lanes),
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    def _prepare_tensor(self, rgb_image):
        """CUDA varsa resize ve normalizasyonu GPU'da, yoksa CPU'da yapar."""
        if not self.cuda_enabled:
            prepared = prepare_ufld_input(rgb_image)
            return self.torch.from_numpy(prepared).to(self.device)

        import torch.nn.functional as functional

        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("UFLD girdisi HxWx3 RGB goruntu olmali.")
        image = np.ascontiguousarray(image[:, :, :3])
        shape = tuple(image.shape)
        if self.host_buffer is None or self.host_buffer_shape != shape:
            self.host_buffer = self.torch.empty(
                shape,
                dtype=self.torch.uint8,
                pin_memory=True,
            )
            self.host_buffer_shape = shape

        self.host_buffer.copy_(self.torch.from_numpy(image))
        tensor = self.host_buffer.to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(
            dtype=(
                self.torch.float16
                if self.use_half
                else self.torch.float32
            )
        )
        tensor = functional.interpolate(
            tensor,
            size=(MODEL_HEIGHT, MODEL_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        tensor = tensor.div_(255.0).sub_(self.mean).div_(self.std)
        return tensor.contiguous(memory_format=self.torch.channels_last)

    def _warmup_cuda(self, dtype):
        """İlk canlı karedeki CUDA kernel hazırlama gecikmesini açılışa taşır."""
        tensor = self.torch.zeros(
            (1, 3, MODEL_HEIGHT, MODEL_WIDTH),
            device=self.device,
            dtype=dtype,
        ).contiguous(memory_format=self.torch.channels_last)
        with self.torch.inference_mode():
            self.model(tensor)
        self.torch.cuda.synchronize()

    def _activate_cuda(self):
        """UFLD modelini CUDA'ya taşır ve ilk inference için hazırlar."""
        self.model = self.model.to(self.device)
        self.model = self.model.to(memory_format=self.torch.channels_last)
        if self.use_half:
            self.model = self.model.half()
        dtype = (
            self.torch.float16
            if self.use_half
            else self.torch.float32
        )
        self.mean = self.torch.tensor(
            IMAGENET_MEAN,
            device=self.device,
            dtype=dtype,
        ).view(1, 3, 1, 1)
        self.std = self.torch.tensor(
            IMAGENET_STD,
            device=self.device,
            dtype=dtype,
        ).view(1, 3, 1, 1)
        self._warmup_cuda(dtype)

    def _fallback_to_cpu(self, error):
        """CUDA belleği yetmediğinde uygulamayı kapatmadan UFLD'yi CPU'ya alır."""
        print(
            f"[WARN] UFLD CUDA bellegi yetersiz ({error}); "
            "serit modeli CPU ile devam edecek."
        )
        self.device = "cpu"
        self.cuda_enabled = False
        self.use_half = False
        self.host_buffer = None
        self.host_buffer_shape = None
        self.mean = None
        self.std = None
        self.model = self.model.to("cpu").float().eval()
        recover_cuda_memory()

    def _is_cuda_memory_error(self, error):
        message = str(error).lower()
        out_of_memory_type = getattr(
            self.torch,
            "OutOfMemoryError",
            RuntimeError,
        )
        return isinstance(error, out_of_memory_type) or (
            "out of memory" in message
            or "cudaerrormemoryallocation" in message
        )
