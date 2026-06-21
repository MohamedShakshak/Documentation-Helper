from abc import ABC, abstractmethod
from typing import Any


class BaseCrawler(ABC):
    @abstractmethod
    async def crawl(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def to_documents(self, raw_results: list[dict[str, Any]]) -> list:
        ...