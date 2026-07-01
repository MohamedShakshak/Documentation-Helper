import re

import httpx
from langchain.tools import tool
from langchain_tavily import TavilySearch

from doc_helper.config.settings import IngestionSettings
from doc_helper.retrieval.retriever import Retriever


def _make_retrieval_tool(retriever: Retriever):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant documentation from the local vector store to help answer
        user queries. Use this tool first before attempting web search."""
        retrieved_docs = retriever.retrieve(query)
        if not retrieved_docs:
            return "No relevant documentation found. Try web_search instead.", []
        serialized = "\n\n".join(
            f"Source: {doc.metadata.get('source_url', doc.metadata.get('source', 'Unknown'))}\n\n"
            f"Content: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


def _make_web_search_tool(tavily_api_key: str | None = None):
    api_key = tavily_api_key
    if not api_key:
        raise ValueError("Tavily API key required for web_search tool")

    client = TavilySearch(tavily_api_key=api_key, max_results=5)

    @tool(response_format="content_and_artifact")
    def web_search(query: str):
        """Search the web for up-to-date information when local documentation
        doesn't contain the answer. Use this for recent API changes, release notes,
        or topics not covered in ingested docs."""
        result = client.invoke({"query": query})
        if isinstance(result, dict):
            results = result.get("results", [])
        elif isinstance(result, list):
            results = result
        else:
            results = []

        snippets: list[str] = []
        urls: list[str] = []
        for item in results[:5]:
            title = item.get("title", "Unknown")
            url = item.get("url", "Unknown")
            content = item.get("content", "")
            snippets.append(f"[{title}]({url})\n{content}")
            urls.append(url)

        if not snippets:
            return "No web results found.", []

        serialized = "\n\n---\n\n".join(snippets)
        return serialized, urls

    return web_search


def _make_check_links_tool():
    @tool
    async def check_links(urls: str):
        """Check whether the provided URLs (newline or comma-separated) are reachable.
        Returns a summary of each URL's HTTP status. Useful for verifying whether
        linked documentation pages are still live."""
        url_list = [u.strip() for u in re.split(r"[\n,]", urls) if u.strip()]
        if not url_list:
            return "No URLs provided."

        results = []
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            for url in url_list:
                try:
                    resp = await client.head(url)
                    results.append(f"{url} -> {resp.status_code}")
                except httpx.ConnectError:
                    results.append(f"{url} -> connection error")
                except httpx.TimeoutException:
                    results.append(f"{url} -> timeout")
                except Exception as e:
                    results.append(f"{url} -> error: {e}")

        return "\n".join(results)

    return check_links


def create_tools(
    retriever: Retriever,
    ingestion_settings: IngestionSettings | None = None,
    enable_web_search: bool = True,
):
    tools = [_make_retrieval_tool(retriever)]

    if enable_web_search:
        settings = ingestion_settings or IngestionSettings()
        if settings.tavily_api_key:
            tools.append(_make_web_search_tool(settings.tavily_api_key))

    tools.append(_make_check_links_tool())
    return tools
