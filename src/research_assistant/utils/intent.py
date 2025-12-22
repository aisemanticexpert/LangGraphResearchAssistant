"""
Figure out what the user is actually asking for.

When someone asks "what's the stock price for Apple?" we should
know to prioritize financial data. When they ask "who runs Tesla?"
we should focus on leadership info. This module does that classification.

Uses regex patterns - not fancy ML, but it's fast and predictable.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """What kind of info is the user after?"""
    NEWS = "news"
    FINANCIAL = "financial"
    DEVELOPMENT = "development"
    COMPETITOR = "competitor"
    LEADERSHIP = "leadership"
    GENERAL = "general"
    COMPARISON = "comparison"


class TimeScope(str, Enum):
    """Are they asking about now, the past, or the future?"""
    CURRENT = "current"
    HISTORICAL = "historical"
    FUTURE = "future"
    UNSPECIFIED = "unspecified"


class DepthLevel(str, Enum):
    """How much detail do they want?"""
    OVERVIEW = "overview"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class IntentAnalysis:
    """Everything we figured out about what the user wants."""

    primary_intent: QueryIntent = QueryIntent.GENERAL
    secondary_intents: List[QueryIntent] = field(default_factory=list)
    time_scope: TimeScope = TimeScope.UNSPECIFIED
    depth_required: DepthLevel = DepthLevel.DETAILED
    specific_aspects: List[str] = field(default_factory=list)
    company_confidence: float = 0.0
    detected_company: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "primary_intent": self.primary_intent.value,
            "secondary_intents": [i.value for i in self.secondary_intents],
            "time_scope": self.time_scope.value,
            "depth_required": self.depth_required.value,
            "specific_aspects": self.specific_aspects,
            "company_confidence": self.company_confidence,
            "detected_company": self.detected_company,
        }

    def get_research_focus(self) -> List[str]:
        """Based on intent, what should we prioritize researching?"""
        focus_mapping = {
            QueryIntent.NEWS: ["recent_news", "announcements", "press_releases"],
            QueryIntent.FINANCIAL: ["stock_info", "revenue", "market_cap", "earnings"],
            QueryIntent.DEVELOPMENT: ["key_developments", "products", "innovations"],
            QueryIntent.COMPETITOR: ["competitors", "market_position", "industry"],
            QueryIntent.LEADERSHIP: ["ceo", "executives", "leadership_changes"],
            QueryIntent.GENERAL: ["recent_news", "stock_info", "key_developments"],
            QueryIntent.COMPARISON: ["competitors", "market_position", "financials"],
        }
        return focus_mapping.get(self.primary_intent, ["recent_news", "stock_info"])


class QueryIntentClassifier:
    """
    Figures out what the user is asking for using keyword matching.

    Nothing fancy here - just regex patterns for common phrases.
    Works well enough for company research queries.
    """

    # words that hint at different intents
    INTENT_PATTERNS = {
        QueryIntent.NEWS: [
            r'\bnews\b', r'\bannounce', r'\bheadlines?\b', r'\bpress\b',
            r'\breport(?:ed|ing)?\b', r'\bupdate\b', r'\blatest\b', r'\brecent\b'
        ],
        QueryIntent.FINANCIAL: [
            r'\bstock\b', r'\bprice\b', r'\bshare\b', r'\btrading\b',
            r'\bmarket\s*cap\b', r'\brevenue\b', r'\bprofit\b', r'\bearnings\b',
            r'\bfinancial\b', r'\bvaluation\b', r'\bdividend\b', r'\binvest'
        ],
        QueryIntent.DEVELOPMENT: [
            r'\bproduct\b', r'\blaunch(?:ed|ing)?\b', r'\bdevelop', r'\binnovation\b',
            r'\brelease\b', r'\bfeature\b', r'\btechnolog', r'\bservice\b'
        ],
        QueryIntent.COMPETITOR: [
            r'\bcompetitor', r'\brival', r'\bvs\.?\b', r'\bversus\b',
            r'\bcompare', r'\bmarket\s*share\b', r'\bcompetition\b'
        ],
        QueryIntent.LEADERSHIP: [
            r'\bceo\b', r'\bchief\b', r'\bexecutive\b', r'\bfounder\b',
            r'\bleader', r'\bmanagement\b', r'\bboard\b', r'\bdirector\b'
        ],
        QueryIntent.COMPARISON: [
            r'\bcompare\b', r'\bdifference\b', r'\bbetter\b', r'\bworse\b',
            r'\badvantage\b', r'\bversus\b', r'\bvs\b'
        ],
    }

    # when are they asking about?
    TIME_PATTERNS = {
        TimeScope.CURRENT: [
            r'\bcurrent\b', r'\bnow\b', r'\btoday\b', r'\blatest\b',
            r'\brecent\b', r'\bthis\s*(week|month|quarter|year)\b'
        ],
        TimeScope.HISTORICAL: [
            r'\bhistor', r'\bpast\b', r'\bprevious\b', r'\blast\s*(week|month|quarter|year)\b',
            r'\b(19|20)\d{2}\b', r'\bago\b', r'\bwas\b', r'\bwere\b'
        ],
        TimeScope.FUTURE: [
            r'\bfuture\b', r'\bupcoming\b', r'\bnext\b', r'\bwill\b',
            r'\bexpect', r'\bforecast', r'\bpredict', r'\bplan'
        ],
    }

    # how much detail do they want?
    DEPTH_PATTERNS = {
        DepthLevel.OVERVIEW: [
            r'\bbrief\b', r'\bquick\b', r'\bsummary\b', r'\boverview\b',
            r'\bshort\b', r'\bsimple\b'
        ],
        DepthLevel.COMPREHENSIVE: [
            r'\bdetail(?:ed)?\b', r'\bcomprehensive\b', r'\bin-?depth\b',
            r'\bthorough\b', r'\bcomplete\b', r'\beverything\b', r'\ball\b'
        ],
    }

    # specific things they might be asking about
    ASPECT_KEYWORDS = {
        "stock_price": [r'\bstock\s*price\b', r'\bshare\s*price\b', r'\btrading\s*at\b'],
        "market_cap": [r'\bmarket\s*cap', r'\bvaluation\b'],
        "revenue": [r'\brevenue\b', r'\bsales\b', r'\bincome\b'],
        "products": [r'\bproduct', r'\bservice', r'\boffering'],
        "competitors": [r'\bcompetitor', r'\brival'],
        "ceo": [r'\bceo\b', r'\bchief\s*executive\b'],
        "headquarters": [r'\bheadquarter', r'\blocation\b', r'\bbased\b'],
        "employees": [r'\bemployee', r'\bstaff\b', r'\bworkforce\b'],
    }

    def __init__(self):
        # compile all the patterns once so we're not doing it every query
        self._compiled_intents = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        self._compiled_time = {
            scope: [re.compile(p, re.IGNORECASE) for p in patterns]
            for scope, patterns in self.TIME_PATTERNS.items()
        }
        self._compiled_depth = {
            depth: [re.compile(p, re.IGNORECASE) for p in patterns]
            for depth, patterns in self.DEPTH_PATTERNS.items()
        }
        self._compiled_aspects = {
            aspect: [re.compile(p, re.IGNORECASE) for p in patterns]
            for aspect, patterns in self.ASPECT_KEYWORDS.items()
        }

    def classify(self, query: str, detected_company: Optional[str] = None) -> IntentAnalysis:
        """
        Take a query, figure out what they're really asking.
        Returns an IntentAnalysis with all the details.
        """
        query_lower = query.lower()

        analysis = IntentAnalysis()
        analysis.detected_company = detected_company

        # figure out primary and secondary intents by counting keyword hits
        intent_scores = self._score_intents(query_lower)
        if intent_scores:
            sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
            analysis.primary_intent = sorted_intents[0][0]
            # secondary intents = anything else with at least one match
            for intent, score in sorted_intents[1:]:
                if score >= 1:
                    analysis.secondary_intents.append(intent)

        analysis.time_scope = self._classify_time_scope(query_lower)
        analysis.depth_required = self._classify_depth(query_lower)
        analysis.specific_aspects = self._extract_aspects(query_lower)
        analysis.company_confidence = self._calculate_company_confidence(
            query, detected_company
        )

        logger.debug(f"Query intent analysis: {analysis.to_dict()}")

        return analysis

    def _score_intents(self, query: str) -> Dict[QueryIntent, int]:
        """Count keyword matches for each intent type."""
        scores = {}
        for intent, patterns in self._compiled_intents.items():
            score = sum(1 for p in patterns if p.search(query))
            if score > 0:
                scores[intent] = score
        return scores

    def _classify_time_scope(self, query: str) -> TimeScope:
        """Past, present, or future?"""
        scope_scores = {}
        for scope, patterns in self._compiled_time.items():
            score = sum(1 for p in patterns if p.search(query))
            if score > 0:
                scope_scores[scope] = score

        if scope_scores:
            return max(scope_scores, key=scope_scores.get)
        return TimeScope.UNSPECIFIED

    def _classify_depth(self, query: str) -> DepthLevel:
        """Brief, normal, or detailed?"""
        for depth, patterns in self._compiled_depth.items():
            if any(p.search(query) for p in patterns):
                return depth
        return DepthLevel.DETAILED  # default to detailed

    def _extract_aspects(self, query: str) -> List[str]:
        """Pull out any specific topics they mentioned."""
        aspects = []
        for aspect, patterns in self._compiled_aspects.items():
            if any(p.search(query) for p in patterns):
                aspects.append(aspect)
        return aspects

    def _calculate_company_confidence(
        self,
        query: str,
        detected_company: Optional[str]
    ) -> float:
        """How sure are we that we got the right company?"""
        if not detected_company:
            return 0.0

        confidence = 0.5  # we detected something, that's worth half

        company_lower = detected_company.lower()
        query_lower = query.lower()

        # full name match is best
        if company_lower in query_lower:
            confidence += 0.3
        # partial match is okay
        elif any(word in query_lower for word in company_lower.split()):
            confidence += 0.15

        # pronouns suggest this is a follow-up about a company we already know
        if re.search(r'\b(the\s+company|them|they|their|it|its)\b', query_lower):
            confidence += 0.1

        return min(1.0, confidence)

    def get_research_strategy(self, analysis: IntentAnalysis) -> Dict[str, any]:
        """Turn the analysis into actual research instructions."""
        return {
            "focus_areas": analysis.get_research_focus(),
            "depth": analysis.depth_required.value,
            "time_emphasis": analysis.time_scope.value,
            "specific_aspects": analysis.specific_aspects,
            "priority_order": self._get_priority_order(analysis),
        }

    def _get_priority_order(self, analysis: IntentAnalysis) -> List[str]:
        """What should we include in the response, in what order?"""
        focus = analysis.get_research_focus()

        # specific asks go first
        priority = list(analysis.specific_aspects)

        # then add the general focus areas
        for area in focus:
            if area not in priority:
                priority.append(area)

        return priority


# reuse the same instance
_classifier_instance: Optional[QueryIntentClassifier] = None


def get_intent_classifier() -> QueryIntentClassifier:
    """Grab the shared classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryIntentClassifier()
    return _classifier_instance


def classify_query_intent(
    query: str,
    detected_company: Optional[str] = None
) -> IntentAnalysis:
    """Quick way to classify a query. Returns IntentAnalysis."""
    classifier = get_intent_classifier()
    return classifier.classify(query, detected_company)
