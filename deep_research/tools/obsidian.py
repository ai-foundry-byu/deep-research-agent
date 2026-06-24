"""Obsidian vault writer for research results"""
import os
from pathlib import Path
from datetime import datetime
import re
import json

def slugify(text: str) -> str:
    """Convert text to URL-friendly slug"""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text[:50]

async def save_to_obsidian(
    query: str,
    report: str,
    iteration_count: int,
    sources_consulted: list,
    confidence_level: str,
    vault_path: str,
    profile: str = "opus",
    cost_summary: dict = None,
) -> str:
    """Save research report to Obsidian vault"""
    try:
        vault_dir = Path(os.path.expanduser(vault_path))
        vault_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        slug = slugify(query)
        filename = f"{date_str}-{profile}-{slug}.md"
        filepath = vault_dir / filename

        cost_str = ""
        if cost_summary:
            cost_str = f"\ncost_usd: {cost_summary.get('total_cost_usd', 0)}"
            cost_str += f"\ntotal_tokens: {cost_summary.get('total_input_tokens', 0) + cost_summary.get('total_output_tokens', 0)}"

        frontmatter = f"""---
query: "{query}"
date: {date_str}
profile: {profile}
iteration_count: {iteration_count}
sources_consulted: {len(sources_consulted)}
confidence_level: {confidence_level}{cost_str}
tags:
  - research
  - agent-generated
  - researcher-{profile}
---

"""
        with open(filepath, 'w') as f:
            f.write(frontmatter)
            f.write(f"# {query}\n\n")
            f.write(f"*Profile: {profile} | Iterations: {iteration_count} | Sources: {len(sources_consulted)}*\n\n")
            if cost_summary:
                f.write(f"*Cost: ${cost_summary.get('total_cost_usd', 0):.4f}*\n\n")
            f.write(report)
            f.write(f"\n\n## Sources\n\n")
            for source in sources_consulted:
                f.write(f"- {source}\n")
            if cost_summary:
                f.write(f"\n\n## Cost Breakdown\n\n")
                f.write(f"| Model | Calls | Cost | Tokens |\n")
                f.write(f"|-------|-------|------|--------|\n")
                for model, stats in cost_summary.get("by_model", {}).items():
                    f.write(f"| {model} | {stats['calls']} | ${stats['cost_usd']:.4f} | {stats['tokens']:,} |\n")

        return str(filepath)
    except Exception as e:
        return f"Error saving to Obsidian: {str(e)}"
