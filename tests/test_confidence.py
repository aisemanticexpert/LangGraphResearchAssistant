"""
Tests for the confidence scoring system.
Makes sure the scores make sense for different inputs.
"""

import pytest
from src.research_assistant.utils.confidence import (
    ConfidenceScorer,
    ConfidenceBreakdown,
    calculate_hybrid_confidence,
    get_confidence_scorer
)


class TestConfidenceBreakdown:
    """Tests for ConfidenceBreakdown dataclass."""

    def test_default_values(self):
        """Should have correct default values."""
        breakdown = ConfidenceBreakdown()
        assert breakdown.data_completeness_score == 0.0
        assert breakdown.data_freshness_score == 0.0
        assert breakdown.query_relevance_score == 0.0
        assert breakdown.data_specificity_score == 0.0
        assert breakdown.source_quality_score == 0.0
        assert breakdown.llm_adjustment == 0.0
        assert breakdown.final_score == 0.0
        assert breakdown.explanations == {}
        assert breakdown.gaps == []

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        breakdown = ConfidenceBreakdown(
            data_completeness_score=8.0,
            final_score=7.5,
            explanations={"test": "value"},
            gaps=["gap1"]
        )
        result = breakdown.to_dict()

        assert "components" in result
        assert "final_score" in result
        assert "explanations" in result
        assert "gaps" in result
        assert result["components"]["data_completeness"] == 8.0
        assert result["final_score"] == 7.5


class TestConfidenceScorerInitialization:
    """Tests for ConfidenceScorer initialization."""

    def test_singleton_instance(self):
        """Should return singleton instance."""
        scorer1 = get_confidence_scorer()
        scorer2 = get_confidence_scorer()
        assert scorer1 is scorer2

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0."""
        scorer = ConfidenceScorer()
        total = sum(scorer.WEIGHTS.values())
        assert abs(total - 1.0) < 0.01

    def test_has_current_year(self):
        """Should have current year set."""
        scorer = ConfidenceScorer()
        assert scorer.current_year > 2020


class TestDataCompletenessScoring:
    """Tests for data completeness scoring component."""

    def test_full_data_high_score(self):
        """Should give high score for complete data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "recent_news": "Company announced new product",
            "stock_info": "Trading at $100",
            "key_developments": "Expanding to new markets",
            "additional_info": {
                "competitors": ["Competitor A"],
                "ceo": "John Doe",
                "industry": "Technology"
            }
        }

        score = scorer._score_data_completeness(findings, breakdown)
        assert score >= 8.0

    def test_partial_data_medium_score(self):
        """Should give medium score for partial data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "recent_news": "Some news",
            "stock_info": None,
            "key_developments": None
        }

        score = scorer._score_data_completeness(findings, breakdown)
        assert 2.0 <= score <= 5.0

    def test_empty_data_low_score(self):
        """Should give low score for empty data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {}

        score = scorer._score_data_completeness(findings, breakdown)
        assert score <= 2.0

    def test_records_missing_fields_as_gaps(self):
        """Should record missing fields as gaps."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {"recent_news": "Some news"}

        scorer._score_data_completeness(findings, breakdown)

        assert "stock information" in breakdown.gaps
        assert "key developments" in breakdown.gaps


class TestDataFreshnessScoring:
    """Tests for data freshness scoring component."""

    def test_current_year_high_score(self):
        """Should give high score for current year data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        current_year = str(scorer.current_year)
        findings = {
            "recent_news": f"In {current_year}, company launched new product"
        }

        score = scorer._score_data_freshness(findings, breakdown)
        assert score >= 5.0

    def test_previous_year_medium_score(self):
        """Should give medium score for previous year data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        prev_year = str(scorer.current_year - 1)
        findings = {
            "recent_news": f"In {prev_year}, company expanded"
        }

        score = scorer._score_data_freshness(findings, breakdown)
        assert score >= 3.0

    def test_freshness_keywords_boost(self):
        """Should boost score for freshness keywords."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "recent_news": "Recently launched new product"
        }

        score = scorer._score_data_freshness(findings, breakdown)
        assert score >= 2.0

    def test_no_date_indicators_low_score(self):
        """Should give low score for no date indicators."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "recent_news": "Company exists"
        }

        score = scorer._score_data_freshness(findings, breakdown)
        assert score <= 3.0


class TestQueryRelevanceScoring:
    """Tests for query relevance scoring component."""

    def test_matching_keywords_high_score(self):
        """Should give high score for matching keywords."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "company_name": "Apple Inc.",
            "stock_info": "Trading at $195",
            "recent_news": "Apple stock hits new high"
        }

        score = scorer._score_query_relevance(findings, "Apple stock price", breakdown)
        assert score >= 6.0

    def test_intent_matching(self):
        """Should boost score for intent matching."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "stock_info": "Trading at $195, up 10%"
        }

        score = scorer._score_query_relevance(findings, "What is the stock price?", breakdown)
        assert score >= 2.0

    def test_no_matching_keywords_low_score(self):
        """Should give low score for no matching keywords."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "recent_news": "Weather is nice today"
        }

        score = scorer._score_query_relevance(findings, "Tesla electric vehicles", breakdown)
        assert score <= 4.0


class TestDataSpecificityScoring:
    """Tests for data specificity scoring component."""

    def test_specific_data_high_score(self):
        """Should give high score for specific data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "company_name": "Apple Inc.",
            "stock_info": "Trading at $195, up 45% YTD",
            "recent_news": "Tim Cook announced Vision Pro launch in January 2024"
        }

        score = scorer._score_data_specificity(findings, breakdown)
        assert score >= 6.0

    def test_numbers_boost_score(self):
        """Should boost score for specific numbers."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "stock_info": "$195 per share, $3 trillion market cap, 25% growth"
        }

        score = scorer._score_data_specificity(findings, breakdown)
        # Numbers contribute to specificity score (at least 1.5 for multiple numbers)
        assert score >= 1.5

    def test_generic_data_low_score(self):
        """Should give low score for generic data."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "company_name": "Unknown Company",
            "recent_news": "company does things"
        }

        score = scorer._score_data_specificity(findings, breakdown)
        assert score <= 4.0

    def test_records_generic_data_as_gap(self):
        """Should record generic data as gap."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "company_name": "Unknown",
            "recent_news": "something happened"
        }

        scorer._score_data_specificity(findings, breakdown)
        assert any("generic" in gap or "specificity" in gap.lower() for gap in breakdown.gaps) or \
               "data_specificity" in breakdown.explanations


class TestSourceQualityScoring:
    """Tests for source quality scoring component."""

    def test_api_source_high_score(self):
        """Should give high score for API source."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "source": "tavily_api",
            "sources": ["tavily", "news_api"],
            "recent_news": "news",
            "stock_info": "stock",
            "key_developments": "dev"
        }

        score = scorer._score_source_quality(findings, breakdown)
        assert score >= 7.0

    def test_mock_source_medium_score(self):
        """Should give medium score for mock source."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "source": "mock_data",
            "sources": ["mock"],
            "recent_news": "news"
        }

        score = scorer._score_source_quality(findings, breakdown)
        assert 4.0 <= score <= 8.0

    def test_multiple_sources_boost(self):
        """Should boost score for multiple sources."""
        scorer = ConfidenceScorer()
        breakdown = ConfidenceBreakdown()

        findings = {
            "sources": ["source1", "source2", "source3"],
            "recent_news": "news",
            "stock_info": "stock"
        }

        score = scorer._score_source_quality(findings, breakdown)
        assert score >= 6.0


class TestHybridConfidenceCalculation:
    """Tests for full hybrid confidence calculation."""

    def test_calculates_final_score(self):
        """Should calculate final hybrid score."""
        findings = {
            "company_name": "Apple Inc.",
            "recent_news": "Vision Pro launched in 2024",
            "stock_info": "Trading at $195",
            "key_developments": "AI integration",
            "source": "mock_data",
            "sources": ["mock"]
        }

        score, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="Tell me about Apple"
        )

        assert 0.0 <= score <= 10.0
        assert isinstance(breakdown, ConfidenceBreakdown)

    def test_llm_score_influences_final(self):
        """Should incorporate LLM score in calculation."""
        findings = {
            "company_name": "Apple Inc.",
            "recent_news": "Some news"
        }

        # With high LLM score
        score_high, _ = calculate_hybrid_confidence(
            findings=findings,
            query="Apple info",
            llm_score=9.0,
            llm_reasoning="Excellent data"
        )

        # With low LLM score
        score_low, _ = calculate_hybrid_confidence(
            findings=findings,
            query="Apple info",
            llm_score=3.0,
            llm_reasoning="Poor data"
        )

        # Higher LLM score should result in higher final score
        assert score_high > score_low

    def test_llm_adjustment_clamped(self):
        """Should clamp LLM adjustment to -2 to +2."""
        findings = {
            "company_name": "Apple Inc.",
            "recent_news": "news"
        }

        _, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="Apple",
            llm_score=10.0  # Very high
        )

        assert -2.0 <= breakdown.llm_adjustment <= 2.0

    def test_score_bounded(self):
        """Should keep final score within 0-10."""
        # Empty findings should still be bounded
        score, _ = calculate_hybrid_confidence(
            findings={},
            query="test"
        )
        assert 0.0 <= score <= 10.0

        # Full findings with high LLM score should be bounded
        score, _ = calculate_hybrid_confidence(
            findings={
                "company_name": "Test",
                "recent_news": "news 2024",
                "stock_info": "$100",
                "key_developments": "dev"
            },
            query="test",
            llm_score=10.0
        )
        assert 0.0 <= score <= 10.0

    def test_complete_data_high_confidence(self):
        """Complete data should yield high confidence."""
        findings = {
            "company_name": "Apple Inc.",
            "recent_news": "Apple launched Vision Pro in January 2024, expanding its product line",
            "stock_info": "Trading at $195, up 45% YTD, market cap $3 trillion",
            "key_developments": "Tim Cook announced AI integration across product line",
            "additional_info": {
                "competitors": ["Microsoft", "Google"],
                "ceo": "Tim Cook",
                "industry": "Technology"
            },
            "source": "tavily_api",
            "sources": ["tavily", "news_api"]
        }

        score, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="Tell me about Apple's latest products and stock price",
            llm_score=8.0
        )

        assert score >= 6.0  # Should be above threshold

    def test_incomplete_data_lower_confidence(self):
        """Incomplete data should yield lower confidence."""
        findings = {
            "company_name": "Unknown Corp",
        }

        score, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="Tell me about Unknown Corp's stock",
            llm_score=3.0
        )

        assert score < 6.0  # Should be below threshold


class TestBreakdownExplanations:
    """Tests for breakdown explanations and gaps."""

    def test_explanations_populated(self):
        """Should populate explanations for each component."""
        findings = {
            "company_name": "Apple Inc.",
            "recent_news": "News from 2024"
        }

        _, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="Apple info"
        )

        # Should have some explanations
        assert len(breakdown.explanations) > 0

    def test_gaps_identified(self):
        """Should identify gaps in data."""
        findings = {
            "company_name": "Apple Inc.",
            # Missing: recent_news, stock_info, key_developments
        }

        _, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="Apple stock price"
        )

        # Should identify missing fields
        assert len(breakdown.gaps) > 0

    def test_llm_reasoning_included(self):
        """Should include LLM reasoning in explanations."""
        findings = {"company_name": "Test"}

        _, breakdown = calculate_hybrid_confidence(
            findings=findings,
            query="test",
            llm_score=7.0,
            llm_reasoning="Good semantic match to query"
        )

        assert "llm_reasoning" in breakdown.explanations
        assert "semantic" in breakdown.explanations["llm_reasoning"].lower()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Should handle empty query."""
        findings = {"company_name": "Apple"}

        score, _ = calculate_hybrid_confidence(
            findings=findings,
            query=""
        )

        assert 0.0 <= score <= 10.0

    def test_empty_findings(self):
        """Should handle empty findings."""
        score, breakdown = calculate_hybrid_confidence(
            findings={},
            query="Tell me about Apple"
        )

        assert score <= 3.0  # Should be low
        assert len(breakdown.gaps) > 0

    def test_none_values_in_findings(self):
        """Should handle None values in findings."""
        findings = {
            "company_name": "Apple",
            "recent_news": None,
            "stock_info": None
        }

        score, _ = calculate_hybrid_confidence(
            findings=findings,
            query="Apple"
        )

        assert 0.0 <= score <= 10.0

    def test_special_characters_in_query(self):
        """Should handle special characters in query."""
        findings = {"company_name": "Apple"}

        score, _ = calculate_hybrid_confidence(
            findings=findings,
            query="What's Apple's stock price? (NYSE: AAPL)"
        )

        assert 0.0 <= score <= 10.0
