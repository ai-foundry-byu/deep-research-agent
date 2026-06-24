"""GitHub search tool"""
import httpx
import os

async def github_search(query: str, count: int = 5) -> str:
    """Search GitHub repositories and code"""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN not set"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search repositories
            response = await client.get(
                "https://api.github.com/search/repositories",
                headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
                params={"q": query, "sort": "stars", "order": "desc", "per_page": count}
            )
            response.raise_for_status()
            results = response.json()
            
            formatted = []
            for repo in results.get("items", []):
                formatted.append(
                    f"**{repo['full_name']}** ⭐ {repo['stargazers_count']}\n"
                    f"{repo['html_url']}\n"
                    f"{repo.get('description', 'No description')}\n"
                    f"Language: {repo.get('language', 'N/A')}, Updated: {repo.get('updated_at', 'N/A')[:10]}\n"
                )
            
            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching GitHub: {str(e)}"
