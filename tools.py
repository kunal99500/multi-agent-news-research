from langchain.tools import tool
from tavily import TavilyClient
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def fetch_web_content(query: str, days: int = 7) -> str:
    """
    Search recent news and return the top 5 results, each tagged with its
    publication date so downstream agents can judge recency.
    """
    results = tavily.search(
        query=query,
        topic="news",        # use Tavily's news index instead of general web
        days=days,           # restrict to the last `days` days
        max_results=5,
        search_depth="advanced",
    )

    docs = []
    for r in results["results"]:
        published = r.get("published_date", "date not stated")
        docs.append(
            f"""
Published: {published}
Title: {r['title']}
URL: {r['url']}
Snippet: {r['content']}
"""
        )

    return "\n---\n".join(docs)




@tool
def scrape_webpage(url: str) -> str:
    """
    Fetch a webpage, extract its publication date (if present) and main text.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # --- try to pull a publication date from common metadata ---
        published = "date not stated"
        meta_keys = [
            ("property", "article:published_time"),
            ("name", "publishdate"),
            ("name", "pubdate"),
            ("itemprop", "datePublished"),
            ("name", "date"),
        ]
        for attr, val in meta_keys:
            tag = soup.find("meta", attrs={attr: val})
            if tag and tag.get("content"):
                published = tag["content"]
                break
        # fallback: a <time datetime="..."> element
        if published == "date not stated":
            t = soup.find("time")
            if t and t.get("datetime"):
                published = t["datetime"]

        # --- strip noise, then extract text ---
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())  # drop blank lines

        return f"Published: {published}\nURL: {url}\n\n{text[:3000]}"

    except requests.exceptions.Timeout:
        return f"Error scraping {url}: request timed out"
    except requests.exceptions.RequestException as e:
        return f"Error scraping {url}: {e}"
    except Exception as e:
        return f"Error scraping {url}: {e}"


