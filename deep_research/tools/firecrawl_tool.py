"""Firecrawl deep page extraction tool"""
import httpx
import os

async def firecrawl_extract(url: str) -> str:
    """Extract full page content via Firecrawl API"""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return f"Error: FIRECRAWL_API_KEY not set. Cannot extract {url}"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.firecrawl.dev/v0/scrape",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"url": url, "pageOptions": {"onlyMainContent": True}}
            )
            response.raise_for_status()
            data = response.json()
            
            content = data.get("data", {}).get("content", "No content extracted")
            # Truncate to avoid huge responses
            if len(content) > 15000:
                content = content[:15000] + "\n\n[Content truncated - total length exceeded 15000 chars]"
            
            return f"**Content from {url}:**\n\n{content}"
    except Exception as e:
        return f"Error extracting {url} via Firecrawl: {str(e)}"
