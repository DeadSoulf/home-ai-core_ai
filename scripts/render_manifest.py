"""Emit a Core Manifest v2 using a real, immutable OCI image digest."""
import argparse
import json
import re
from pathlib import Path


def render(digest: str) -> dict:
    digest = digest.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Provide a real 64-hex sha256 image digest")
    path = Path(__file__).resolve().parent.parent / "manifest.template.json"
    return json.loads(path.read_text().replace("IMAGE_DIGEST", digest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("digest", help="Published OCI image digest (sha256:...) from GHCR")
    args = parser.parse_args()
    print(json.dumps(render(args.digest), indent=2, ensure_ascii=False))
