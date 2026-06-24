"""CLI interface for research agent — supports opus and multi profiles.

Phase 4: Added --no-cache, --format, --notify, --output flags.
"""
import asyncio
import argparse
import json
import os
import sys
from pathlib import Path

from deep_research.core.loop import run_research
from deep_research.core.progress import ProgressReporter
from deep_research.core.formatter import format_report, VALID_FORMATS


def main():
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - Iterative knowledge gap research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  opus   — Premium: Opus orchestrator + Sonnet workers (best quality)
  multi  — Budget: Sonnet orchestrator + Gemini Flash/Ollama workers (cheapest)

Formats:
  markdown — Default Markdown report
  html     — Styled HTML for email/sharing
  pdf      — PDF via HTML conversion (requires weasyprint or wkhtmltopdf)
  json     — Structured JSON for programmatic use

Examples:
  %(prog)s "best multi-agent orchestration patterns 2026"
  %(prog)s "LangGraph vs CrewAI comparison" --profile multi --depth deep
  %(prog)s "latest developments in autonomous agents" --notify --format html
  %(prog)s "AI safety research" --no-cache --output ~/reports/safety.md
        """
    )

    parser.add_argument("query", nargs="?", default=None, help="Research query")
    parser.add_argument(
        "--profile",
        choices=["opus", "multi"],
        default="opus",
        help="Model profile: opus (premium — routes via the Claude Code CLI / Claude SDK) or multi (budget — OpenRouter cloud + local Ollama). Default: opus"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum research iterations (overrides depth preset)"
    )
    parser.add_argument(
        "--max-time",
        type=int,
        help="Maximum time in minutes (default: 15)"
    )
    parser.add_argument(
        "--depth",
        choices=["shallow", "standard", "deep"],
        default="standard",
        help="Research depth: shallow (2 iter), standard (5 iter), deep (10 iter)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint if one exists for this query+depth"
    )

    # Phase 4 flags
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip result cache — always run fresh research"
    )
    parser.add_argument(
        "--format",
        choices=list(VALID_FORMATS),
        default="markdown",
        dest="output_format",
        help="Report output format (default: markdown)"
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send real-time progress updates to Telegram"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output file path (in addition to the default reports directory)"
    )
    parser.add_argument(
        "--verify-claims",
        dest="verify_claims",
        action="store_true",
        default=None,
        help="Force per-claim adversarial verification ON (overrides depth gate)"
    )
    parser.add_argument(
        "--no-verify-claims",
        dest="verify_claims",
        action="store_false",
        help="Force per-claim adversarial verification OFF (overrides depth gate)"
    )

    # --- chat-v2 --reply contract ---
    parser.add_argument("--reply", type=str, default=None,
                        help="Chat message (unified chat addressee contract)")
    parser.add_argument("--reply-file", type=str, default=None,
                        help="Read message from file (for long inputs)")
    parser.add_argument("--thread-id", type=str, default=None,
                        help="Chat thread ID (ignored — researcher is stateless)")
    parser.add_argument("--file", dest="reply_files", action="append", default=[],
                        help="File attachment (ignored by researcher)")
    parser.add_argument("--reply-dry-run", action="store_true",
                        help="Echo a stub reply without running research (wiring test)")

    args = parser.parse_args()

    # --- chat-v2 --reply path: fold into the query slot ---
    chat_mode = False
    if args.reply or args.reply_file:
        chat_mode = True
        if args.reply_file:
            args.query = Path(args.reply_file).read_text(encoding="utf-8")
        else:
            args.query = args.reply

        if args.reply_dry_run:
            print(f"stub reply received: {args.query[:200]}")
            sys.exit(0)

        # chat-mode defaults (cost + latency guards)
        args.depth = "shallow"
        args.notify = False
        args.output_format = "markdown"
        if args.reply_files:
            print(f"(researcher ignores --file attachments; got {len(args.reply_files)})", file=sys.stderr)

    if not args.query:
        parser.error("query is required (or pass --reply)")

    # Setup progress reporter
    progress = ProgressReporter(enabled=args.notify)

    try:
        result = asyncio.run(run_research(
            query=args.query,
            max_iterations=args.max_iterations,
            max_time_minutes=args.max_time,
            depth=args.depth,
            profile=args.profile,
            resume=args.resume,
            progress=progress,
            use_cache=not args.no_cache,
            output_format=args.output_format,
            output_path=args.output,
            verify_claims=args.verify_claims,
        ))

        if chat_mode or os.environ.get("CHAT_V2_CONTEXT") == "1":
            # Plain markdown — no banner noise — plus an artifact event
            print(result.report)
            if result.obsidian_path:
                print("[[EVENT]] " + json.dumps({
                    "type": "artifact",
                    "path": str(result.obsidian_path),
                    "label": "Full research report",
                    "mime": "text/markdown",
                }))
        else:
            print("\n" + "=" * 80)
            print(f"RESEARCH REPORT  [{args.profile.upper()} profile]")
            print("=" * 80 + "\n")
            print(result.report)
            print("\n" + "=" * 80)
            print(f"Profile:    {result.profile}")
            print(f"Saved to:   {result.obsidian_path}")
            print(f"Iterations: {result.iteration_count}")
            print(f"Sources:    {len(result.sources_consulted)}")
            print(f"Cost:       ${result.cost_summary.get('total_cost_usd', 0):.4f}")
            if args.output_format != "markdown":
                print(f"Format:     {args.output_format}")
            if args.output:
                print(f"Output:     {args.output}")
            print("=" * 80 + "\n")

        # If custom output path and format is markdown, also save there
        if args.output and args.output_format == "markdown":
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(result.report, encoding="utf-8")
            print(f"Also saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
