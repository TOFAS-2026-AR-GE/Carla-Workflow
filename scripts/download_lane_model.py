"""Doğrulanmış Hugging Face deposundan UFLD CARLA ağırlığını indirir."""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "lane" / "ufld_carla_best.pth"
REPOSITORY = "jkdxbns/autonomous-driving-carla"
FILENAME = "ufld_carla_best.pth"
EXPECTED_SHA256 = "c8880d9e2fb42615cca8f15faeb4ee4c88f22b8519ced94d2e5a827a1cc689ec"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Var olan dosyayı Hugging Face üzerinden yeniden indir.",
    )
    arguments = parser.parse_args()

    if TARGET.is_file() and not arguments.force:
        if sha256(TARGET) == EXPECTED_SHA256:
            print(f"[OK] UFLD modeli zaten var ve SHA256 dogru: {TARGET}")
            return 0
        print("[WARN] Var olan UFLD dosyasinin SHA256 degeri uyusmuyor.")
        arguments.force = True

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] Once 'pip install huggingface-hub' calistirin.")
        return 1

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPOSITORY,
        filename=FILENAME,
        local_dir=TARGET.parent,
        force_download=arguments.force,
    )
    downloaded = Path(path)
    if sha256(downloaded) != EXPECTED_SHA256:
        print("[ERROR] Indirilen UFLD modelinin SHA256 degeri gecersiz.")
        return 1
    print(f"[OK] UFLD modeli indirildi ve dogrulandi: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
