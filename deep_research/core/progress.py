"""Telegram streaming progress for research runs.

Sends real-time progress updates to Telegram during research.
Rate-limited to max 1 update per 30 seconds to avoid spam.
Uses the existing telegram_notify.py pattern (Bot API via urllib).
"""
import asyncio
import time
import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

# Optional Telegram progress streaming. Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
# in the environment to enable; otherwise progress updates are silently skipped.
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Minimum interval between updates (seconds)
MIN_UPDATE_INTERVAL = 30


@dataclass
class ProgressReporter:
    """Sends rate-limited progress updates to Telegram.

    Usage:
        progress = ProgressReporter(enabled=True)
        await progress.update("Searching Brave...", emoji="🔍")
        await progress.update("Found 15 sources...", emoji="📚")
        await progress.complete("Research complete!", report_path="/path/to/report.md")
    """

    enabled: bool = False
    chat_id: str = field(default_factory=lambda: CHAT_ID)
    _last_update_time: float = 0.0
    _message_id: Optional[int] = None  # Track message for edits
    _pending_text: Optional[str] = None
    _query: str = ""
    _start_time: float = field(default_factory=time.time)

    def set_query(self, query: str):
        """Set the research query for inclusion in status messages."""
        self._query = query
        self._start_time = time.time()

    async def update(self, status: str, emoji: str = "🔄", force: bool = False):
        """Send a progress update to Telegram.

        Rate-limited: updates are batched and sent at most once per MIN_UPDATE_INTERVAL.
        Use force=True to bypass rate limiting (for critical updates like start/end).
        """
        if not self.enabled or not BOT_TOKEN:
            return

        now = time.time()
        elapsed = int(now - self._start_time)
        elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s" if elapsed >= 60 else f"{elapsed}s"

        text = f"{emoji} {status}\n⏱ {elapsed_str}"
        if self._query:
            text = f"📋 *{_escape_md(self._query)}*\n{text}"

        if not force and (now - self._last_update_time) < MIN_UPDATE_INTERVAL:
            # Queue update — it will be sent on next allowed interval or on force
            self._pending_text = text
            return

        # Send or edit
        await self._send_or_edit(text)
        self._last_update_time = now
        self._pending_text = None

    async def flush(self):
        """Send any pending update that was rate-limited."""
        if self._pending_text and self.enabled:
            await self._send_or_edit(self._pending_text)
            self._pending_text = None
            self._last_update_time = time.time()

    async def complete(self, summary: str, report_path: str = "", cost_usd: float = 0.0):
        """Send a completion notification (always sends, bypasses rate limit).

        Sends a NEW message (not an edit) so the user gets a push notification.
        """
        if not self.enabled or not BOT_TOKEN:
            return

        elapsed = int(time.time() - self._start_time)
        elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s" if elapsed >= 60 else f"{elapsed}s"

        text = f"✅ *Research Complete*\n"
        if self._query:
            text += f"📋 {_escape_md(self._query)}\n"
        text += f"\n{summary}\n"
        text += f"⏱ {elapsed_str}"
        if cost_usd > 0:
            text += f" | 💰 ${cost_usd:.4f}"
        if report_path:
            text += f"\n📁 Saved to vault"

        # Always send as new message for push notification
        await self._send_new(text)

    async def error(self, error_msg: str):
        """Send an error notification (always sends)."""
        if not self.enabled or not BOT_TOKEN:
            return

        text = f"❌ *Research Failed*\n"
        if self._query:
            text += f"📋 {_escape_md(self._query)}\n"
        text += f"\n{_escape_md(error_msg)}"

        await self._send_new(text)

    async def _send_or_edit(self, text: str):
        """Edit the existing progress message, or send a new one."""
        if self._message_id:
            success = await asyncio.to_thread(
                _edit_message, self.chat_id, self._message_id, text
            )
            if success:
                return
            # Edit failed — fall through to send new

        msg_id = await asyncio.to_thread(
            _send_message, self.chat_id, text
        )
        if msg_id:
            self._message_id = msg_id

    async def _send_new(self, text: str):
        """Always send a new message (for completion notifications)."""
        await asyncio.to_thread(_send_message, self.chat_id, text)


def _escape_md(text: str) -> str:
    """Escape special MarkdownV2 characters for Telegram.

    We use regular Markdown parse_mode, so only escape the most problematic chars.
    """
    # For regular Markdown mode, underscores and brackets need escaping
    return text.replace("_", "\\_").replace("[", "\\[").replace("]", "\\]")


def _send_message(chat_id: str, text: str) -> Optional[int]:
    """Send a new Telegram message. Returns message_id on success."""
    if not BOT_TOKEN:
        return None

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text[:4000],
        "parse_mode": "Markdown",
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                return result["result"]["message_id"]
    except Exception as e:
        print(f"   [telegram] Send failed: {e}")
    return None


def _edit_message(chat_id: str, message_id: int, text: str) -> bool:
    """Edit an existing Telegram message."""
    if not BOT_TOKEN:
        return False

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text[:4000],
        "parse_mode": "Markdown",
    }

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            return result.get("ok", False)
    except Exception as e:
        # Common: "message is not modified" — not an error
        if "not modified" in str(e).lower():
            return True
        print(f"   [telegram] Edit failed: {e}")
        return False
