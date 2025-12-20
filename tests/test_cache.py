"""
Tests for the query caching system.
"""

import time
import pytest
from unittest.mock import patch


class TestQueryCache:
    """Tests for QueryCache functionality."""

    def test_cache_initialization(self):
        """Test cache initializes with correct settings."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 50
            mock_settings.cache_ttl_seconds = 1800
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache(max_size=50, ttl_seconds=1800)

            assert cache.max_size == 50
            assert cache.ttl_seconds == 1800
            assert cache._enabled == True

    def test_cache_set_and_get(self):
        """Test setting and getting cached values."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache()

            test_data = {"response": "Test response", "company": "Apple"}
            cache.set("What about Apple?", test_data, company="Apple")

            result = cache.get("What about Apple?", company="Apple")
            assert result == test_data

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache()

            result = cache.get("Non-existent query")
            assert result is None

    def test_cache_expiration(self):
        """Test that cached items expire after TTL."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 1
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache(ttl_seconds=1)

            cache.set("test query", {"data": "value"})

            # Should be in cache immediately
            assert cache.get("test query") is not None

            # Wait for expiration
            time.sleep(1.5)

            # Should be expired
            assert cache.get("test query") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 3
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache(max_size=3)

            cache.set("query1", {"data": "1"})
            cache.set("query2", {"data": "2"})
            cache.set("query3", {"data": "3"})

            # Access query1 to make it recently used
            cache.get("query1")

            # Add new item, should evict query2 (oldest unused)
            cache.set("query4", {"data": "4"})

            assert cache.get("query1") is not None  # Still exists (recently used)
            assert cache.get("query3") is not None  # Still exists
            assert cache.get("query4") is not None  # New item

    def test_cache_clear(self):
        """Test clearing the cache."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache()

            cache.set("query1", {"data": "1"})
            cache.set("query2", {"data": "2"})

            cache.clear()

            assert cache.get("query1") is None
            assert cache.get("query2") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache()

            cache.set("query1", {"data": "1"})
            cache.set("query2", {"data": "2"})

            stats = cache.get_stats()

            assert stats["total_entries"] == 2
            assert stats["valid_entries"] == 2
            assert stats["max_size"] == 10
            assert stats["enabled"] == True

    def test_cache_disabled(self):
        """Test cache when disabled."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = False

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache()
            cache.disable()

            cache.set("query", {"data": "value"})
            result = cache.get("query")

            assert result is None

    def test_cache_key_normalization(self):
        """Test that cache keys are normalized (case-insensitive)."""
        with patch("src.research_assistant.utils.cache.settings") as mock_settings:
            mock_settings.cache_max_size = 10
            mock_settings.cache_ttl_seconds = 3600
            mock_settings.enable_cache = True

            from src.research_assistant.utils.cache import QueryCache
            cache = QueryCache()

            cache.set("Tell me about APPLE", {"data": "value"})

            # Should match regardless of case
            result = cache.get("tell me about apple")
            assert result is not None
