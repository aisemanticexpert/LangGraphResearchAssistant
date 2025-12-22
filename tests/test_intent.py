"""
Tests for intent classification.
Checks that we correctly figure out what users are asking for.
"""

import pytest
from src.research_assistant.utils.intent import (
    QueryIntentClassifier,
    QueryIntent,
    TimeScope,
    DepthLevel,
    IntentAnalysis,
    classify_query_intent,
    get_intent_classifier,
)


class TestQueryIntentClassifier:
    """Tests for QueryIntentClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance."""
        return QueryIntentClassifier()

    def test_news_intent_detection(self, classifier):
        """Should detect news intent."""
        analysis = classifier.classify("What's the latest news about Apple?")
        assert analysis.primary_intent == QueryIntent.NEWS

    def test_financial_intent_detection(self, classifier):
        """Should detect financial intent."""
        analysis = classifier.classify("What is Tesla's stock price?")
        assert analysis.primary_intent == QueryIntent.FINANCIAL

    def test_development_intent_detection(self, classifier):
        """Should detect development/product intent."""
        analysis = classifier.classify("What new products has Google launched?")
        assert analysis.primary_intent == QueryIntent.DEVELOPMENT

    def test_competitor_intent_detection(self, classifier):
        """Should detect competitor analysis intent."""
        analysis = classifier.classify("Who are Microsoft's main competitors?")
        assert analysis.primary_intent == QueryIntent.COMPETITOR

    def test_leadership_intent_detection(self, classifier):
        """Should detect leadership intent."""
        analysis = classifier.classify("Who is the CEO of Amazon?")
        assert analysis.primary_intent == QueryIntent.LEADERSHIP

    def test_comparison_intent_detection(self, classifier):
        """Should detect comparison intent."""
        analysis = classifier.classify("How does Apple compare to Samsung?")
        # Compare triggers both COMPARISON and COMPETITOR patterns
        # Either is acceptable for a comparison query
        assert analysis.primary_intent in [QueryIntent.COMPARISON, QueryIntent.COMPETITOR]

    def test_general_intent_default(self, classifier):
        """Should default to general for vague queries."""
        analysis = classifier.classify("Tell me about Netflix")
        assert analysis.primary_intent == QueryIntent.GENERAL

    def test_secondary_intents_detected(self, classifier):
        """Should detect secondary intents."""
        analysis = classifier.classify("What's Apple's stock price and latest news?")
        assert QueryIntent.FINANCIAL in [analysis.primary_intent] + analysis.secondary_intents
        assert QueryIntent.NEWS in [analysis.primary_intent] + analysis.secondary_intents


class TestTimeScopeClassification:
    """Tests for time scope classification."""

    @pytest.fixture
    def classifier(self):
        return QueryIntentClassifier()

    def test_current_time_scope(self, classifier):
        """Should detect current time scope."""
        analysis = classifier.classify("What's Apple's current stock price?")
        assert analysis.time_scope == TimeScope.CURRENT

    def test_historical_time_scope(self, classifier):
        """Should detect historical time scope."""
        analysis = classifier.classify("What was Tesla's revenue last year?")
        assert analysis.time_scope == TimeScope.HISTORICAL

    def test_future_time_scope(self, classifier):
        """Should detect future time scope."""
        analysis = classifier.classify("What's Google's expected revenue growth?")
        assert analysis.time_scope == TimeScope.FUTURE

    def test_unspecified_time_scope(self, classifier):
        """Should default to unspecified for ambiguous queries."""
        analysis = classifier.classify("Tell me about Microsoft")
        assert analysis.time_scope == TimeScope.UNSPECIFIED


class TestDepthClassification:
    """Tests for depth level classification."""

    @pytest.fixture
    def classifier(self):
        return QueryIntentClassifier()

    def test_overview_depth(self, classifier):
        """Should detect overview/brief depth."""
        analysis = classifier.classify("Give me a brief summary of Apple")
        assert analysis.depth_required == DepthLevel.OVERVIEW

    def test_comprehensive_depth(self, classifier):
        """Should detect comprehensive depth."""
        analysis = classifier.classify("I need a detailed analysis of Tesla's business")
        assert analysis.depth_required == DepthLevel.COMPREHENSIVE

    def test_default_detailed_depth(self, classifier):
        """Should default to detailed depth."""
        analysis = classifier.classify("Tell me about Google's products")
        assert analysis.depth_required == DepthLevel.DETAILED


class TestSpecificAspectsExtraction:
    """Tests for specific aspect extraction."""

    @pytest.fixture
    def classifier(self):
        return QueryIntentClassifier()

    def test_stock_price_aspect(self, classifier):
        """Should extract stock price aspect."""
        analysis = classifier.classify("What is Apple's stock price?")
        assert "stock_price" in analysis.specific_aspects

    def test_ceo_aspect(self, classifier):
        """Should extract CEO aspect."""
        analysis = classifier.classify("Who is the CEO of Microsoft?")
        assert "ceo" in analysis.specific_aspects

    def test_competitors_aspect(self, classifier):
        """Should extract competitors aspect."""
        analysis = classifier.classify("Who are Google's competitors?")
        assert "competitors" in analysis.specific_aspects

    def test_multiple_aspects(self, classifier):
        """Should extract multiple aspects."""
        analysis = classifier.classify("What is Tesla's stock price and who is the CEO?")
        assert "stock_price" in analysis.specific_aspects
        assert "ceo" in analysis.specific_aspects


class TestCompanyConfidence:
    """Tests for company identification confidence."""

    @pytest.fixture
    def classifier(self):
        return QueryIntentClassifier()

    def test_no_company_zero_confidence(self, classifier):
        """Should have zero confidence when no company detected."""
        analysis = classifier.classify("What is the stock price?")
        assert analysis.company_confidence == 0.0

    def test_company_detected_has_confidence(self, classifier):
        """Should have confidence when company is detected."""
        analysis = classifier.classify("Tell me about Apple", detected_company="Apple Inc.")
        assert analysis.company_confidence > 0.0

    def test_company_in_query_higher_confidence(self, classifier):
        """Should have higher confidence when company appears in query."""
        analysis = classifier.classify("Tell me about Apple", detected_company="Apple")
        assert analysis.company_confidence >= 0.8


class TestResearchFocus:
    """Tests for research focus generation."""

    def test_news_focus(self):
        """Should generate news-focused research priorities."""
        analysis = IntentAnalysis(primary_intent=QueryIntent.NEWS)
        focus = analysis.get_research_focus()
        assert "recent_news" in focus

    def test_financial_focus(self):
        """Should generate financial-focused research priorities."""
        analysis = IntentAnalysis(primary_intent=QueryIntent.FINANCIAL)
        focus = analysis.get_research_focus()
        assert "stock_info" in focus

    def test_leadership_focus(self):
        """Should generate leadership-focused research priorities."""
        analysis = IntentAnalysis(primary_intent=QueryIntent.LEADERSHIP)
        focus = analysis.get_research_focus()
        assert "ceo" in focus


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_get_intent_classifier_singleton(self):
        """Should return singleton instance."""
        classifier1 = get_intent_classifier()
        classifier2 = get_intent_classifier()
        assert classifier1 is classifier2

    def test_classify_query_intent_function(self):
        """Should classify using convenience function."""
        analysis = classify_query_intent("What is Apple's stock price?")
        assert analysis.primary_intent == QueryIntent.FINANCIAL


class TestIntentAnalysisToDict:
    """Tests for IntentAnalysis serialization."""

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        analysis = IntentAnalysis(
            primary_intent=QueryIntent.FINANCIAL,
            secondary_intents=[QueryIntent.NEWS],
            time_scope=TimeScope.CURRENT,
            depth_required=DepthLevel.DETAILED,
            specific_aspects=["stock_price"],
            company_confidence=0.9,
            detected_company="Apple Inc."
        )

        result = analysis.to_dict()

        assert result["primary_intent"] == "financial"
        assert result["secondary_intents"] == ["news"]
        assert result["time_scope"] == "current"
        assert result["depth_required"] == "detailed"
        assert result["specific_aspects"] == ["stock_price"]
        assert result["company_confidence"] == 0.9
        assert result["detected_company"] == "Apple Inc."
