"""Scheduled research jobs — run recurring research topics via cron.

Reads job definitions from a YAML config file. Each job specifies a query,
depth, schedule, and output path. When triggered, runs research and saves
the report to the configured vault location.

Usage (from cron):
    python -m deep_research.core.scheduler

Or run a specific job:
    python -m deep_research.core.scheduler --job "ai-frameworks"

Config file: ./research_schedules.yaml (override with --config)
"""
import asyncio
import os
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

# Default config path
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("DEEP_RESEARCH_SCHEDULE_CONFIG", "research_schedules.yaml")
)

# Default output base path (override with DEEP_RESEARCH_OUTPUT_DIR).
DEFAULT_OUTPUT_BASE = Path(os.path.expanduser(
    os.environ.get("DEEP_RESEARCH_OUTPUT_DIR", "~/deep-research-reports") + "/scheduled"
))


@dataclass
class ScheduledJob:
    """A scheduled research job definition."""
    name: str
    query: str
    depth: str = "standard"
    profile: str = "opus"
    schedule: str = "weekly"  # cron expression or shorthand: daily, weekly, monthly
    output_path: str = ""
    notify: bool = True
    enabled: bool = True

    @property
    def cron_expression(self) -> str:
        """Convert schedule shorthand to cron expression."""
        shortcuts = {
            "daily": "0 6 * * *",       # 6 AM daily
            "weekly": "0 6 * * 1",      # 6 AM Monday
            "monthly": "0 6 1 * *",     # 6 AM 1st of month
        }
        return shortcuts.get(self.schedule, self.schedule)


def load_jobs(config_path: Optional[Path] = None) -> list[ScheduledJob]:
    """Load scheduled jobs from YAML config."""
    if yaml is None:
        raise ImportError("PyYAML required for scheduled jobs: pip install pyyaml")

    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        # Create example config
        _create_example_config(path)
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data or "jobs" not in data:
        return []

    jobs = []
    for job_data in data["jobs"]:
        jobs.append(ScheduledJob(
            name=job_data.get("name", "unnamed"),
            query=job_data["query"],
            depth=job_data.get("depth", "standard"),
            profile=job_data.get("profile", "opus"),
            schedule=job_data.get("schedule", "weekly"),
            output_path=job_data.get("output_path", ""),
            notify=job_data.get("notify", True),
            enabled=job_data.get("enabled", True),
        ))

    return [j for j in jobs if j.enabled]


def _create_example_config(path: Path):
    """Create an example config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    example = """# Research Agent — Scheduled Jobs
# Each job runs research on a recurring schedule and saves to the vault.
#
# Schedule options:
#   - "daily"    → 6 AM every day
#   - "weekly"   → 6 AM every Monday
#   - "monthly"  → 6 AM 1st of month
#   - Or a cron expression: "0 8 * * 3" (8 AM Wednesday)
#
# Add to crontab:
#   0 6 * * 1 cd /path/to/deep-research-agent && python -m deep_research.core.scheduler >> scheduler.log 2>&1

jobs:
  - name: ai-frameworks
    query: "latest AI agent frameworks and orchestration tools 2026"
    depth: standard
    profile: opus
    schedule: weekly
    notify: true
    enabled: true

  - name: llm-developments
    query: "major LLM developments and releases this week"
    depth: shallow
    profile: multi
    schedule: weekly
    notify: true
    enabled: false

  - name: autonomous-agents
    query: "autonomous AI agent research breakthroughs"
    depth: deep
    profile: opus
    schedule: monthly
    notify: true
    enabled: false
"""
    path.write_text(example)
    print(f"[scheduler] Created example config: {path}")


async def run_job(job: ScheduledJob) -> dict:
    """Run a single scheduled research job.

    Returns a dict with job name, status, report_path, and cost.
    """
    from ..core.loop import run_research
    from ..core.progress import ProgressReporter

    print(f"\n{'='*60}")
    print(f"[scheduler] Running job: {job.name}")
    print(f"[scheduler] Query: {job.query}")
    print(f"[scheduler] Depth: {job.depth}, Profile: {job.profile}")
    print(f"{'='*60}\n")

    progress = ProgressReporter(enabled=job.notify)
    progress.set_query(f"[Scheduled] {job.query}")

    try:
        await progress.update("Starting scheduled research...", emoji="📅", force=True)

        result = await run_research(
            query=job.query,
            depth=job.depth,
            profile=job.profile,
            progress=progress,
        )

        # Save to custom output path if specified
        report_path = result.obsidian_path
        if job.output_path:
            output_dir = Path(os.path.expanduser(job.output_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            output_file = output_dir / f"{date_str}-{job.name}.md"
            # Copy report to custom location
            if Path(report_path).exists():
                import shutil
                shutil.copy2(report_path, output_file)
                report_path = str(output_file)

        cost = result.cost_summary.get("total_cost_usd", 0)

        await progress.complete(
            summary=(
                f"📊 {result.iteration_count} iterations, "
                f"{len(result.sources_consulted)} sources\n"
                f"🎯 Confidence: {result.confidence_level}"
            ),
            report_path=report_path,
            cost_usd=cost,
        )

        return {
            "job": job.name,
            "status": "complete",
            "report_path": report_path,
            "iterations": result.iteration_count,
            "sources": len(result.sources_consulted),
            "confidence": result.confidence_level,
            "cost_usd": cost,
        }

    except Exception as e:
        await progress.error(f"Job '{job.name}' failed: {str(e)[:200]}")
        return {
            "job": job.name,
            "status": "error",
            "error": str(e),
        }


async def run_all_due_jobs(config_path: Optional[Path] = None, job_filter: Optional[str] = None):
    """Run all enabled jobs (or a specific job by name).

    In production, cron handles scheduling — this function just runs
    whatever jobs are enabled when called.
    """
    jobs = load_jobs(config_path)

    if not jobs:
        print("[scheduler] No enabled jobs found")
        return []

    if job_filter:
        jobs = [j for j in jobs if j.name == job_filter]
        if not jobs:
            print(f"[scheduler] Job '{job_filter}' not found or not enabled")
            return []

    results = []
    for job in jobs:
        result = await run_job(job)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"[scheduler] Completed {len(results)} job(s)")
    for r in results:
        status_icon = "✅" if r["status"] == "complete" else "❌"
        print(f"  {status_icon} {r['job']}: {r['status']}")
    print(f"{'='*60}\n")

    return results


def main():
    """CLI entry point for scheduler."""
    import argparse

    parser = argparse.ArgumentParser(description="Run scheduled research jobs")
    parser.add_argument("--job", help="Run a specific job by name")
    parser.add_argument("--config", help="Path to config YAML")
    parser.add_argument("--list", action="store_true", help="List configured jobs")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None

    if args.list:
        jobs = load_jobs(config_path)
        if not jobs:
            print("No enabled jobs.")
            return
        for j in jobs:
            status = "enabled" if j.enabled else "disabled"
            print(f"  [{status}] {j.name}: {j.query} ({j.depth}, {j.schedule})")
        return

    asyncio.run(run_all_due_jobs(config_path, job_filter=args.job))


if __name__ == "__main__":
    main()
