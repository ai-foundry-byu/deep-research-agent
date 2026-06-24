"""Report format options — Markdown, HTML, PDF, and structured JSON.

Converts the default Markdown report into alternative formats for
sharing, email, and programmatic use.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


VALID_FORMATS = ("markdown", "html", "pdf", "json")


@dataclass
class FormattedReport:
    """A report in the requested format."""
    content: str  # The formatted content (or file path for PDF)
    format: str
    file_path: Optional[str] = None  # Path if saved to disk
    content_type: str = "text/markdown"


def format_report(
    report_markdown: str,
    format: str = "markdown",
    query: str = "",
    metadata: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> FormattedReport:
    """Convert a Markdown report to the requested format.

    Args:
        report_markdown: The report in Markdown format (default output).
        format: One of "markdown", "html", "pdf", "json".
        query: The original research query (for metadata).
        metadata: Additional metadata dict (iterations, sources, cost, etc.).
        output_path: If provided, save the formatted report to this path.

    Returns:
        FormattedReport with the content in the requested format.
    """
    if format not in VALID_FORMATS:
        raise ValueError(f"Unknown format '{format}'. Must be one of {VALID_FORMATS}")

    if format == "markdown":
        result = FormattedReport(
            content=report_markdown,
            format="markdown",
            content_type="text/markdown",
        )

    elif format == "html":
        html = _markdown_to_html(report_markdown, query)
        result = FormattedReport(
            content=html,
            format="html",
            content_type="text/html",
        )

    elif format == "pdf":
        html = _markdown_to_html(report_markdown, query)
        pdf_path = _html_to_pdf(html, output_path)
        result = FormattedReport(
            content=pdf_path,  # For PDF, content is the file path
            format="pdf",
            file_path=pdf_path,
            content_type="application/pdf",
        )

    elif format == "json":
        structured = _to_structured_json(report_markdown, query, metadata)
        result = FormattedReport(
            content=json.dumps(structured, indent=2, ensure_ascii=False),
            format="json",
            content_type="application/json",
        )

    # Save to disk if output_path is provided (for non-PDF formats)
    if output_path and format != "pdf":
        ext_map = {"markdown": ".md", "html": ".html", "json": ".json"}
        ext = ext_map.get(format, ".txt")
        path = output_path if output_path.endswith(ext) else output_path + ext
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(result.content, encoding="utf-8")
        result.file_path = path

    return result


def _markdown_to_html(markdown_text: str, query: str = "") -> str:
    """Convert Markdown to HTML with styling.

    Uses a simple regex-based converter to avoid heavy dependencies.
    For production use, consider installing `markdown` or `mistune`.
    """
    try:
        import markdown as md_lib
        body = md_lib.markdown(
            markdown_text,
            extensions=["tables", "fenced_code", "toc"],
        )
    except ImportError:
        # Fallback: basic regex conversion
        body = _basic_md_to_html(markdown_text)

    title = query or "Research Report"
    date = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{_html_escape(title)}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    line-height: 1.6;
    color: #333;
    background: #fafafa;
  }}
  h1 {{ color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 0.5rem; }}
  h2 {{ color: #16213e; margin-top: 2rem; }}
  h3 {{ color: #0f3460; }}
  a {{ color: #e94560; }}
  code {{ background: #f0f0f0; padding: 0.2em 0.4em; border-radius: 3px; font-size: 0.9em; }}
  pre {{ background: #2d2d2d; color: #f0f0f0; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
  pre code {{ background: none; color: inherit; }}
  blockquote {{ border-left: 4px solid #e94560; margin-left: 0; padding-left: 1rem; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: left; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 2rem; }}
  .confidence-high {{ color: #27ae60; font-weight: bold; }}
  .confidence-medium {{ color: #f39c12; font-weight: bold; }}
  .confidence-low {{ color: #e74c3c; font-weight: bold; }}
</style>
</head>
<body>
<div class="meta">Generated {date} by Research Agent</div>
{body}
</body>
</html>"""


def _basic_md_to_html(text: str) -> str:
    """Basic Markdown to HTML conversion (no external dependencies)."""
    lines = text.split("\n")
    html_lines = []
    in_list = False
    in_code_block = False

    for line in lines:
        # Code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                html_lines.append("</code></pre>")
                in_code_block = False
            else:
                html_lines.append("<pre><code>")
                in_code_block = True
            continue

        if in_code_block:
            html_lines.append(_html_escape(line))
            continue

        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{_inline_md(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{_inline_md(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{_inline_md(line[4:])}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{_inline_md(line[5:])}</h4>")
        # Bullet lists
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            content = line.strip()[2:]
            html_lines.append(f"<li>{_inline_md(content)}</li>")
        # Horizontal rules
        elif line.strip() in ("---", "***", "___"):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("<hr>")
        # Empty lines
        elif not line.strip():
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
        # Regular paragraphs
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{_inline_md(line)}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    """Convert inline Markdown (bold, italic, code, links)."""
    text = _html_escape(text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # Links
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
    # Confidence markers
    text = text.replace("[high confidence]", '<span class="confidence-high">[high confidence]</span>')
    text = text.replace("[medium confidence]", '<span class="confidence-medium">[medium confidence]</span>')
    text = text.replace("[low confidence]", '<span class="confidence-low">[low confidence]</span>')
    return text


def _html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _html_to_pdf(html: str, output_path: Optional[str] = None) -> str:
    """Convert HTML to PDF.

    Tries weasyprint first, then falls back to writing HTML with a note.
    """
    if not output_path:
        output_path = f"/tmp/research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    # Ensure .pdf extension
    if not output_path.endswith(".pdf"):
        output_path = output_path.rsplit(".", 1)[0] + ".pdf"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(output_path)
        return output_path
    except ImportError:
        pass

    # Fallback: use wkhtmltopdf if available
    try:
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            f.write(html)
            html_path = f.name

        result = subprocess.run(
            ["wkhtmltopdf", "--quiet", html_path, output_path],
            capture_output=True,
            timeout=30,
        )
        Path(html_path).unlink(missing_ok=True)

        if result.returncode == 0:
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Last resort: save as HTML with .pdf.html extension
    fallback_path = output_path + ".html"
    Path(fallback_path).write_text(html, encoding="utf-8")
    print(f"   [formatter] PDF conversion unavailable. Install weasyprint or wkhtmltopdf.")
    print(f"   [formatter] Saved HTML version: {fallback_path}")
    return fallback_path


def _to_structured_json(
    markdown_text: str,
    query: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """Convert a Markdown report to structured JSON for programmatic use."""
    sections = _extract_sections(markdown_text)

    result = {
        "query": query,
        "generated_at": datetime.now().isoformat(),
        "format_version": "1.0",
        "report": {
            "title": query,
            "sections": sections,
            "full_markdown": markdown_text,
        },
    }

    if metadata:
        result["metadata"] = metadata

    return result


def _extract_sections(markdown_text: str) -> list[dict]:
    """Extract sections from Markdown by splitting on headers."""
    sections = []
    current_section = None
    current_content = []

    for line in markdown_text.split("\n"):
        header_match = re.match(r'^(#{1,4})\s+(.+)', line)
        if header_match:
            # Save previous section
            if current_section:
                current_section["content"] = "\n".join(current_content).strip()
                sections.append(current_section)

            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_section = {
                "level": level,
                "title": title,
                "content": "",
            }
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_section:
        current_section["content"] = "\n".join(current_content).strip()
        sections.append(current_section)

    return sections
