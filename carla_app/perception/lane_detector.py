"""CARLA için klasik UFLD şerit algılama adaptörü.

Kaynak model: https://huggingface.co/jkdxbns/autonomous-driving-carla
Özgün decode: https://github.com/cfzd/Ultra-Fast-Lane-Detection
"""

import time
from pathlib import Path

import cv2
import numpy as np

from carla_app.perception.device import is_cuda_device, resolve_device

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
        use_half=True,
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
        self.cuda_enabled = is_cuda_device(self.device)
        self.use_half = bool(use_half and self.cuda_enabled)
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
        self.host_buffer = None
        self.host_buffer_shape = None
        if self.cuda_enabled:
            self.model = self.model.to(memory_format=torch.channels_last)
            if self.use_half:
                self.model = self.model.half()
            dtype = torch.float16 if self.use_half else torch.float32
            self.mean = torch.tensor(
                IMAGENET_MEAN,
                device=self.device,
                dtype=dtype,
            ).view(1, 3, 1, 1)
            self.std = torch.tensor(
                IMAGENET_STD,
                device=self.device,
                dtype=dtype,
            ).view(1, 3, 1, 1)
            self._warmup_cuda(dtype)
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
