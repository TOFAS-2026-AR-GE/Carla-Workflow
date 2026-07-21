"""CARLA için klasik UFLD şerit algılama adaptörü.

Kaynak model: https://huggingface.co/jkdxbns/autonomous-driving-carla
Özgün decode: https://github.com/cfzd/Ultra-Fast-Lane-Detection
"""

import time
from pathlib import Path

import cv2
import numpy as np

from carla_app.perception.device import resolve_device

MODEL_WIDTH = 800
MODEL_HEIGHT = 288
GRID_SIZE = 100
ROW_COUNT = 56
LANE_COUNT = 4
ROW_ANCHORS = np.linspace(115, 287, ROW_COUNT).astype(np.int32)
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


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


class LaneDetector:
    """Ön RGB kamerada UFLD çıkarımı yapar; kontrol komutu üretmez."""

    def __init__(
        self,
        model_path,
        device="auto",
        minimum_points=3,
        minimum_confidence=0.30,
    ):
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"UFLD model dosyasi bulunamadi: {path}")

        import torch

        from carla_app.perception.ufld_model import build_ufld_resnet18

        requested_device = resolve_device(device)
        if str(requested_device).isdigit():
            requested_device = f"cuda:{requested_device}"
        self.device = str(requested_device)
        self.minimum_points = max(3, int(minimum_points))
        self.minimum_confidence = float(minimum_confidence)
        self.torch = torch

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

        self.model = model.to(self.device).eval()

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
        prepared = prepare_ufld_input(rgb_image)
        tensor = self.torch.from_numpy(prepared).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(tensor)
        logits = logits.detach().float().cpu().numpy()
        height, width = rgb_image.shape[:2]
        lanes = decode_ufld_logits(
            logits,
            image_width=width,
            image_height=height,
            minimum_points=self.minimum_points,
            minimum_confidence=self.minimum_confidence,
        )
        return {
            "available": True,
            "model": "ufld_resnet18_carla",
            "model_input_size": [MODEL_WIDTH, MODEL_HEIGHT],
            "image_size": [int(width), int(height)],
            "lanes": lanes,
            "detected_count": sum(lane["detected"] for lane in lanes),
            "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
        }
