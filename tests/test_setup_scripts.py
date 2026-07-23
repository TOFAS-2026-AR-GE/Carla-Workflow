import hashlib
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import check_setup, download_lane_model


def settings(vehicle_model, lane_model=None):
    return types.SimpleNamespace(
        vehicle_model=vehicle_model,
        enable_sign_detection=False,
        enable_lane_detection=lane_model is not None,
        lane_model=lane_model,
        vehicle_device="auto",
        sign_device="auto",
        lane_device="auto",
        enable_fp16_inference=True,
    )


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class SetupCheckTests(unittest.TestCase):
    def test_missing_package_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "vehicle.pt"
            model.write_bytes(b"model")

            def finder(package):
                return None if package == "cv2" else object()

            with redirect_stdout(StringIO()) as output:
                result = check_setup.main(
                    settings=settings(model),
                    package_finder=finder,
                    module_importer=lambda _name: types.SimpleNamespace(
                        cuda=FakeCuda(),
                    ),
                )

        self.assertEqual(result, 1)
        self.assertIn("EKSIK cv2", output.getvalue())

    def test_missing_enabled_lane_model_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            vehicle_model = Path(directory) / "vehicle.pt"
            vehicle_model.write_bytes(b"model")
            lane_model = Path(directory) / "missing-lane.pth"

            with redirect_stdout(StringIO()) as output:
                result = check_setup.main(
                    settings=settings(vehicle_model, lane_model),
                    package_finder=lambda _package: object(),
                    module_importer=lambda _name: types.SimpleNamespace(
                        cuda=FakeCuda(),
                    ),
                )

        self.assertEqual(result, 1)
        self.assertIn(f"EKSIK {lane_model}", output.getvalue())

    def test_complete_cpu_setup_returns_success(self):
        with tempfile.TemporaryDirectory() as directory:
            vehicle_model = Path(directory) / "vehicle.pt"
            lane_model = Path(directory) / "lane.pth"
            vehicle_model.write_bytes(b"vehicle")
            lane_model.write_bytes(b"lane")

            with redirect_stdout(StringIO()):
                result = check_setup.main(
                    settings=settings(vehicle_model, lane_model),
                    package_finder=lambda _package: object(),
                    module_importer=lambda _name: types.SimpleNamespace(
                        cuda=FakeCuda(),
                    ),
                )

        self.assertEqual(result, 0)


class LaneModelDownloadTests(unittest.TestCase):
    def test_download_is_pinned_and_hash_verified(self):
        payload = b"verified lane model"
        expected_hash = hashlib.sha256(payload).hexdigest()
        captured = {}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ufld_carla_best.pth"

            def fake_download(**arguments):
                captured.update(arguments)
                target.write_bytes(payload)
                return str(target)

            huggingface_hub = types.ModuleType("huggingface_hub")
            huggingface_hub.hf_hub_download = fake_download
            with (
                patch.object(download_lane_model, "TARGET", target),
                patch.object(
                    download_lane_model,
                    "EXPECTED_SHA256",
                    expected_hash,
                ),
                patch.dict(
                    sys.modules,
                    {"huggingface_hub": huggingface_hub},
                ),
                redirect_stdout(StringIO()),
            ):
                result = download_lane_model.main([])

        self.assertEqual(result, 0)
        self.assertEqual(
            captured["revision"],
            download_lane_model.REVISION,
        )
        self.assertEqual(captured["repo_id"], download_lane_model.REPOSITORY)
        self.assertFalse(captured["force_download"])


if __name__ == "__main__":
    unittest.main()
