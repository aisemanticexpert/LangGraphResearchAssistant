"""
Query caching for the Research Assistant.

Provides in-memory caching with TTL (time-to-live) support to avoid
redundant API calls for identical queries.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, Optional

from ..config import settings

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Simple LRU cache with TTL for query results.

    Caches research results to avoid redundant API calls when users
    ask similar questions. Uses an OrderedDict for LRU eviction.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        ttl_seconds: Optional[int] = None
    ):
        self.max_size = max_size or settings.cache_max_size
        self.ttl_seconds = ttl_seconds or settings.cache_ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._enabled = settings.enable_cache
        logger.info(f"Cache initialized (max_size={self.max_size}, ttl={self.ttl_seconds}s)")

    def _generate_key(self, query: str, company: Optional[str] = None) -> str:
        """Generate a cache key from query and company name."""
        normalized = query.lower().strip()
        if company:
            normalized += f"|{company.lower().strip()}"
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str, company: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached result for a query.

        Args:
            query: The user query
            company: Optional company name for more specific caching

        Returns:
            Cached result dict or None if not found/expired
        """
        if not self._enabled:
            return None

        key = self._generate_key(query, company)

        if key not in self._cache:
            logger.debug(f"Cache miss: {query[:50]}...")
            return None

        entry = self._cache[key]

        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            logger.debug(f"Cache expired: {query[:50]}...")
            del self._cache[key]
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        logger.info(f"Cache hit: {query[:50]}...")
        return entry["data"]

    def set(
        self,
        query: str,
        data: Dict[str, Any],
        company: Optional[str] = None
    ) -> None:
        """
        Store a result in cache.

        Args:
            query: The user query
            data: The result data to cache
            company: Optional company name
        """
        if not self._enabled:
            return

        key = self._generate_key(query, company)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug("Evicted oldest cache entry")

        self._cache[key] = {
            "data": data,
            "timestamp": time.time(),
            "query": query[:100],  # Store truncated query for debugging
        }
        logger.debug(f"Cached: {query[:50]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        valid_count = sum(
            1 for entry in self._cache.values()
            if now - entry["timestamp"] <= self.ttl_seconds
        )
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "enabled": self._enabled,
        }

    def enable(self) -> None:
        """Enable caching."""
        self._enabled = True
        logger.info("Cache enabled")

    def disable(self) -> None:
        """Disable caching."""
        self._enabled = False
        logger.info("Cache disabled")


# Global cache instance
query_cache = QueryCache()
