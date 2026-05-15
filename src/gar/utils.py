from __future__ import annotations

import hashlib
import re
from pathlib import Path


SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def safe_collection_name(name: str) -> str:
    cleaned = SAFE_NAME_PATTERN.sub("-", name.strip()).strip("-").lower()
    return cleaned or "default"


def tokenize_for_sparse(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def short_snippet(text: str, limit: int = 320) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
