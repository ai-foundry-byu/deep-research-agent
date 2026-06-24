"""MCP server for Claude Code integration — exposes both researcher profiles."""
import asyncio
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_research.core.loop import run_research


TOOLS = [
    {
        "name": "research_opus",
        "description": (
            "Premium deep research using Claude Opus orchestrator + Sonnet workers. "
            "Best quality, higher cost (~$0.50-2.00/report). "
            "Searches web, GitHub, Reddit, arXiv, HN. Returns comprehensive report with citations."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research query or question"},
                "depth": {
                    "type": "string",
                    "enum": ["shallow", "standard", "deep"],
                    "description": "shallow=2 iterations, standard=5, deep=10",
                    "default": "standard",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "research_multi",
        "description": (
            "Budget deep research using Sonnet orchestrator + Gemini Flash/Ollama workers. "
            "Good quality, very cheap (~$0.02-0.10/report). "
            "Same search sources as opus. Uses local models where possible."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research query or question"},
                "depth": {
                    "type": "string",
                    "enum": ["shallow", "standard", "deep"],
                    "description": "shallow=2 iterations, standard=5, deep=10",
                    "default": "standard",
                },
            },
            "required": ["query"],
        },
    },
]


async def handle_research(args: dict, profile: str) -> dict:
    """Run research and return MCP-formatted result."""
    query = args.get("query", "")
    depth = args.get("depth", "standard")

    if not query:
        return {"content": [{"type": "text", "text": "Error: No query provided"}], "isError": True}

    try:
        result = await run_research(query=query, depth=depth, profile=profile)

        cost = result.cost_summary
        cost_text = f"${cost.get('total_cost_usd', 0):.4f}" if cost else "unknown"

        return {
            "content": [
                {"type": "text", "text": f"# Research Report: {query}\n\n{result.report}"},
                {
                    "type": "text",
                    "text": (
                        f"\n\n---\n\n**Metadata:**\n"
                        f"- Profile: {profile}\n"
                        f"- Iterations: {result.iteration_count}\n"
                        f"- Sources: {len(result.sources_consulted)}\n"
                        f"- Cost: {cost_text}\n"
                        f"- Saved to: {result.obsidian_path}"
                    ),
                },
            ]
        }
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Error during research: {e}"}], "isError": True}


async def main():
    """MCP stdio server — JSON-RPC over stdin/stdout."""
    print("🔬 Research Agent MCP Server v2.0 starting...", file=sys.stderr)
    print("   Tools: research_opus, research_multi", file=sys.stderr)

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        try:
            line = await reader.readline()
            if not line:
                break

            request = json.loads(line.decode())
            method = request.get("method", "")
            req_id = request.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "research-agent", "version": "2.0.0"},
                        "capabilities": {"tools": {}},
                    },
                }
            elif method == "notifications/initialized":
                continue  # no response needed
            elif method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
            elif method == "tools/call":
                tool_name = request.get("params", {}).get("name", "")
                args = request.get("params", {}).get("arguments", {})

                if tool_name == "research_opus":
                    result = await handle_research(args, "opus")
                elif tool_name == "research_multi":
                    result = await handle_research(args, "multi")
                else:
                    result = {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if req_id:
                err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())

# Register in Claude Code:
# claude mcp add research-agent "python3.12 /path/to/deep-research-agent/deep_research/mcp_server.py"
