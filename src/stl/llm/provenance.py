"""Small provenance helpers shared by the active LLM campaigns."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any


def digest_json(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_digest(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_digest(path: str | Path) -> str:
    """Digest the saved adapter files that define one LoRA checkpoint."""
    files = sorted(Path(path).glob("adapter*"))
    if not any(file.suffix == ".safetensors" for file in files):
        raise FileNotFoundError(f"no saved adapter: {path}")
    return digest_json([
        (file.name, file_digest(file)) for file in files if file.is_file()
    ])
