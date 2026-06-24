"""Core checkpoint manager — save, load, list, delete, resume."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .checkpoint_models import Checkpoint, CheckpointInfo, CheckpointMetadata, SCHEMA_VERSION

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR = Path(
    os.environ.get("DEEP_RESEARCH_CHECKPOINT_DIR")
    or str(Path.home() / ".deep-research" / "checkpoints")
)


class CheckpointManager:
    """Manages saving, loading, listing, and deleting workflow checkpoints.

    Checkpoints are stored as JSON files under:
        {base_dir}/{task_id}/{stage}_{timestamp}.json

    All writes are atomic (write to .tmp, then rename) and protected by
    file-level locking via fcntl.

    Usage:
        mgr = CheckpointManager()
        mgr.save("research-42", "iteration-3", {"findings": [...]})
        state = mgr.load("research-42")
        mgr.delete("research-42", stage="iteration-1")
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_CHECKPOINT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def save(
        self,
        task_id: str,
        stage: str,
        state_dict: dict[str, Any],
        *,
        agent_name: str | None = None,
        description: str | None = None,
        parent_checkpoint: str | None = None,
    ) -> Path:
        """Atomically save a checkpoint and return its path.

        Writes to a temporary file first, then renames into place so that
        a crash mid-write never leaves a corrupted checkpoint.
        """
        now = datetime.now(timezone.utc)
        ts_str = now.isoformat()
        ts_file = now.strftime("%Y%m%dT%H%M%S_%f")

        task_dir = self.base_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{stage}_{ts_file}.json"
        target = task_dir / filename

        metadata = CheckpointMetadata(
            task_id=task_id,
            stage=stage,
            created_at=ts_str,
            schema_version=SCHEMA_VERSION,
            agent_name=agent_name,
            description=description,
            parent_checkpoint=parent_checkpoint,
        )
        checkpoint = Checkpoint(metadata=metadata, data=state_dict)
        payload = checkpoint.to_json(pretty=True)

        # Atomic write: write to temp file in the same directory, then rename.
        fd, tmp_path = tempfile.mkstemp(
            dir=str(task_dir), prefix=".ckpt_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f, fcntl.LOCK_UN)
            os.rename(tmp_path, str(target))
        except BaseException:
            # Clean up temp file on any failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info("Saved checkpoint %s/%s (%d bytes)", task_id, stage, len(payload))
        return target

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def load(
        self, task_id: str, stage: str | None = None
    ) -> dict[str, Any] | None:
        """Load the latest checkpoint for a task, optionally filtered by stage.

        Returns the state dict, or None if no matching checkpoint exists.
        On parse errors, logs a warning and returns None.
        """
        task_dir = self.base_dir / task_id
        if not task_dir.is_dir():
            return None

        candidates = self._find_checkpoints(task_dir, stage=stage)
        if not candidates:
            return None

        # Most recent first (sorted by filename which embeds timestamp).
        candidates.sort(key=lambda p: p.name, reverse=True)
        best = candidates[0]

        try:
            with open(best, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                raw = f.read()
                fcntl.flock(f, fcntl.LOCK_UN)
            cp = Checkpoint.from_json(raw)

            if cp.metadata.schema_version != SCHEMA_VERSION:
                logger.warning(
                    "Schema version mismatch for %s: expected %s, got %s. "
                    "Ignoring checkpoint.",
                    best,
                    SCHEMA_VERSION,
                    cp.metadata.schema_version,
                )
                return None

            logger.info("Loaded checkpoint %s", best)
            return cp.data
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupted checkpoint %s: %s", best, exc)
            return None

    def load_full(
        self, task_id: str, stage: str | None = None
    ) -> Checkpoint | None:
        """Like load(), but returns the full Checkpoint (metadata + data)."""
        task_dir = self.base_dir / task_id
        if not task_dir.is_dir():
            return None

        candidates = self._find_checkpoints(task_dir, stage=stage)
        if not candidates:
            return None

        candidates.sort(key=lambda p: p.name, reverse=True)
        best = candidates[0]

        try:
            with open(best, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                raw = f.read()
                fcntl.flock(f, fcntl.LOCK_UN)
            return Checkpoint.from_json(raw)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupted checkpoint %s: %s", best, exc)
            return None

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list(self, task_id: str | None = None) -> list[CheckpointInfo]:
        """List checkpoints, optionally filtered to a single task.

        Returns a list of CheckpointInfo sorted by timestamp descending.
        """
        results: list[CheckpointInfo] = []

        if task_id:
            task_dirs = [self.base_dir / task_id]
        else:
            task_dirs = [
                d for d in self.base_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

        for task_dir in task_dirs:
            if not task_dir.is_dir():
                continue
            for path in task_dir.glob("*.json"):
                info = self._path_to_info(path)
                if info:
                    results.append(info)

        results.sort(key=lambda i: i.timestamp, reverse=True)
        return results

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete(self, task_id: str, stage: str | None = None) -> int:
        """Delete checkpoints for a task. Returns number of files removed.

        If stage is given, only checkpoints matching that stage are deleted.
        If stage is None, all checkpoints for the task are removed (including
        the task directory itself).
        """
        task_dir = self.base_dir / task_id
        if not task_dir.is_dir():
            return 0

        targets = self._find_checkpoints(task_dir, stage=stage)
        removed = 0
        for path in targets:
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", path, exc)

        # Remove the task directory if it is now empty.
        try:
            if task_dir.is_dir() and not any(task_dir.iterdir()):
                task_dir.rmdir()
        except OSError:
            pass

        logger.info("Deleted %d checkpoint(s) for %s/%s", removed, task_id, stage or "*")
        return removed

    # ------------------------------------------------------------------
    # resume_or_start
    # ------------------------------------------------------------------

    def resume_or_start(
        self,
        task_id: str,
        initial_state: dict[str, Any],
        stage: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Load the latest checkpoint if one exists, else return initial_state.

        Returns:
            (state_dict, resumed_from) where resumed_from is the checkpoint
            filename if resumed, or None if starting fresh.
        """
        task_dir = self.base_dir / task_id
        if task_dir.is_dir():
            candidates = self._find_checkpoints(task_dir, stage=stage)
            if candidates:
                candidates.sort(key=lambda p: p.name, reverse=True)
                best = candidates[0]
                try:
                    with open(best, "r") as f:
                        fcntl.flock(f, fcntl.LOCK_SH)
                        raw = f.read()
                        fcntl.flock(f, fcntl.LOCK_UN)
                    cp = Checkpoint.from_json(raw)
                    if cp.metadata.schema_version == SCHEMA_VERSION:
                        logger.info("Resuming task %s from %s", task_id, best.name)
                        return cp.data, best.name
                    else:
                        logger.warning(
                            "Schema mismatch on resume for %s; starting fresh.", task_id
                        )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Corrupted checkpoint on resume for %s: %s; starting fresh.",
                        task_id,
                        exc,
                    )

        logger.info("Starting task %s fresh (no checkpoint found).", task_id)
        return initial_state, None

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_checkpoints(
        task_dir: Path, stage: str | None = None
    ) -> list[Path]:
        """Return checkpoint .json files in a task directory, optionally filtered."""
        if stage:
            return list(task_dir.glob(f"{stage}_*.json"))
        return list(task_dir.glob("*.json"))

    @staticmethod
    def _path_to_info(path: Path) -> CheckpointInfo | None:
        """Parse a checkpoint filename into a CheckpointInfo."""
        name = path.stem  # e.g. "iteration-3_20260406T120000_000000"
        # Split on the last two underscore-separated timestamp parts.
        parts = name.rsplit("_", 2)
        if len(parts) < 2:
            return None

        # Stage is everything before the timestamp portion.
        # Timestamp format: YYYYMMDDTHHMMSS_ffffff  (two parts after rsplit by 2)
        if len(parts) == 3:
            stage = parts[0]
            ts_raw = parts[1] + "_" + parts[2]
        else:
            stage = parts[0]
            ts_raw = parts[1]

        # Convert filename timestamp back to ISO format for display.
        try:
            dt = datetime.strptime(ts_raw, "%Y%m%dT%H%M%S_%f")
            dt = dt.replace(tzinfo=timezone.utc)
            ts_iso = dt.isoformat()
        except ValueError:
            ts_iso = ts_raw

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        return CheckpointInfo(
            task_id=path.parent.name,
            stage=stage,
            timestamp=ts_iso,
            size_bytes=size,
            path=str(path),
        )
