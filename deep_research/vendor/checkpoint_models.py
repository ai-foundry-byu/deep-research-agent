"""Data models for the checkpoint system."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1.0"


@dataclass
class CheckpointInfo:
    """Summary info about a checkpoint file on disk."""

    task_id: str
    stage: str
    timestamp: str  # ISO 8601
    size_bytes: int
    path: str

    @property
    def created_at(self) -> datetime:
        return datetime.fromisoformat(self.timestamp)

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


@dataclass
class CheckpointMetadata:
    """Metadata stored inside a checkpoint file."""

    task_id: str
    stage: str
    created_at: str  # ISO 8601
    schema_version: str = SCHEMA_VERSION
    agent_name: Optional[str] = None
    description: Optional[str] = None
    parent_checkpoint: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """A full checkpoint: metadata + arbitrary state data."""

    metadata: CheckpointMetadata
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "data": self.data,
        }

    def to_json(self, pretty: bool = True) -> str:
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Checkpoint:
        meta_dict = d["metadata"]
        metadata = CheckpointMetadata(**meta_dict)
        return cls(metadata=metadata, data=d.get("data", {}))

    @classmethod
    def from_json(cls, text: str) -> Checkpoint:
        return cls.from_dict(json.loads(text))
