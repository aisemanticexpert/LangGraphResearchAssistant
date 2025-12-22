"""
Hybrid confidence scoring - mixes hard rules with LLM judgment.

The idea: LLMs are great at nuance but terrible at consistency.
Rules are consistent but miss context. So we do both.

60% comes from rule-based checks (testable, predictable)
40% comes from LLM assessment (catches the weird edge cases)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """
    All the pieces that make up a confidence score.
    Useful for debugging why a score came out the way it did.
    """

    # each of these is 0-10
    data_completeness_score: float = 0.0
    data_freshness_score: float = 0.0
    query_relevance_score: float = 0.0
    data_specificity_score: float = 0.0
    source_quality_score: float = 0.0

    # LLM can nudge the score up or down a bit (-2 to +2)
    llm_adjustment: float = 0.0

    final_score: float = 0.0

    # why each component scored the way it did
    explanations: Dict[str, str] = field(default_factory=dict)

    # things we noticed were missing
    gaps: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Dump everything to a dict for logging."""
        return {
            "components": {
                "data_completeness": round(self.data_completeness_score, 2),
                "data_freshness": round(self.data_freshness_score, 2),
                "query_relevance": round(self.query_relevance_score, 2),
                "data_specificity": round(self.data_specificity_score, 2),
                "source_quality": round(self.source_quality_score, 2),
            },
            "llm_adjustment": round(self.llm_adjustment, 2),
            "final_score": round(self.final_score, 2),
            "explanations": self.explanations,
            "gaps": self.gaps
        }


class ConfidenceScorer:
    """
    The main scoring engine.

    Weights were tuned by hand based on what felt right during testing.
    Feel free to tweak them if your use case is different.
    """

    # how much each factor matters
    WEIGHTS = {
        "data_completeness": 0.25,  # do we have all the fields?
        "data_freshness": 0.15,     # is the data recent?
        "query_relevance": 0.25,    # does it answer the question?
        "data_specificity": 0.20,   # concrete facts vs vague fluff
        "source_quality": 0.15,     # where'd this come from?
    }

    RULE_BASED_WEIGHT = 0.6  # rules get the majority vote
    LLM_WEIGHT = 0.4         # LLM gets to adjust

    def __init__(self):
        self.current_year = datetime.now().year

    def calculate_confidence(
        self,
        findings: Dict[str, Any],
        query: str,
        llm_score: Optional[float] = None,
        llm_reasoning: Optional[str] = None
    ) -> ConfidenceBreakdown:
        """
        Run all the checks and combine into a final score.

        Pass in llm_score if you already got one from the LLM.
        Otherwise it'll just use the rule-based score.
        """
        breakdown = ConfidenceBreakdown()

        # run each check
        breakdown.data_completeness_score = self._score_data_completeness(findings, breakdown)
        breakdown.data_freshness_score = self._score_data_freshness(findings, breakdown)
        breakdown.query_relevance_score = self._score_query_relevance(findings, query, breakdown)
        breakdown.data_specificity_score = self._score_data_specificity(findings, breakdown)
        breakdown.source_quality_score = self._score_source_quality(findings, breakdown)

        # weighted average of all components
        rule_based_score = (
            breakdown.data_completeness_score * self.WEIGHTS["data_completeness"] +
            breakdown.data_freshness_score * self.WEIGHTS["data_freshness"] +
            breakdown.query_relevance_score * self.WEIGHTS["query_relevance"] +
            breakdown.data_specificity_score * self.WEIGHTS["data_specificity"] +
            breakdown.source_quality_score * self.WEIGHTS["source_quality"]
        )

        # let LLM nudge it if we have a score from them
        if llm_score is not None:
            # cap the adjustment so LLM can't go crazy
            raw_adjustment = (llm_score - rule_based_score) * 0.5
            breakdown.llm_adjustment = max(-2.0, min(2.0, raw_adjustment))

            if llm_reasoning:
                breakdown.explanations["llm_reasoning"] = llm_reasoning

        # blend rule-based and LLM-adjusted scores
        breakdown.final_score = (
            rule_based_score * self.RULE_BASED_WEIGHT +
            (rule_based_score + breakdown.llm_adjustment) * self.LLM_WEIGHT
        )

        # keep it in bounds
        breakdown.final_score = max(0.0, min(10.0, breakdown.final_score))

        logger.debug(f"Confidence breakdown: {breakdown.to_dict()}")

        return breakdown

    def _score_data_completeness(
        self,
        findings: Dict[str, Any],
        breakdown: ConfidenceBreakdown
    ) -> float:
        """
        Do we have all the expected fields filled in?
        Each main field is worth 2.5 points, up to 10 total.
        """
        score = 0.0
        missing = []

        # the big three
        if findings.get("recent_news"):
            score += 2.5
        else:
            missing.append("recent news")

        if findings.get("stock_info"):
            score += 2.5
        else:
            missing.append("stock information")

        if findings.get("key_developments"):
            score += 2.5
        else:
            missing.append("key developments")

        # bonus points for the extras
        additional = findings.get("additional_info", {})
        if not additional:
            additional = findings.get("raw_data", {}).get("additional_info", {})

        additional_score = 0.0
        if additional.get("competitors"):
            additional_score += 0.75
        if additional.get("ceo"):
            additional_score += 0.5
        if additional.get("industry"):
            additional_score += 0.5
        if additional.get("headquarters"):
            additional_score += 0.25
        if additional.get("founded"):
            additional_score += 0.25
        if additional.get("employees"):
            additional_score += 0.25

        score += min(2.5, additional_score)

        if missing:
            breakdown.gaps.extend(missing)
            breakdown.explanations["data_completeness"] = f"Missing: {', '.join(missing)}"
        else:
            breakdown.explanations["data_completeness"] = "All key data fields present"

        return min(10.0, score)

    def _score_data_freshness(
        self,
        findings: Dict[str, Any],
        breakdown: ConfidenceBreakdown
    ) -> float:
        """
        How recent is this data? We look for year mentions,
        temporal keywords like "recent" or "just announced",
        and penalize if we see old years mentioned.
        """
        score = 0.0
        findings_text = str(findings).lower()

        current_year = str(self.current_year)
        previous_year = str(self.current_year - 1)

        # current year = good, last year = okay
        if current_year in findings_text:
            score += 5.0
            breakdown.explanations["data_freshness"] = f"Contains {current_year} data"
        elif previous_year in findings_text:
            score += 3.0
            breakdown.explanations["data_freshness"] = f"Contains {previous_year} data"

        # words that suggest recent info
        freshness_keywords = ["recent", "latest", "new", "just", "announced", "launched", "q1", "q2", "q3", "q4"]
        if any(kw in findings_text for kw in freshness_keywords):
            score += 2.0

        # bonus for specific dates
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b\d{1,2}/\d{4}\b',
            r'\bq[1-4]\s*\d{4}\b'
        ]
        for pattern in date_patterns:
            if re.search(pattern, findings_text):
                score += 1.0
                break

        # ding if we see old years (pre-2024 or whatever)
        outdated_years = [str(y) for y in range(2019, self.current_year - 1)]
        if any(year in findings_text for year in outdated_years):
            score -= 1.0
            breakdown.gaps.append("contains potentially outdated information")

        if "data_freshness" not in breakdown.explanations:
            if score >= 5:
                breakdown.explanations["data_freshness"] = "Data appears current"
            elif score >= 2:
                breakdown.explanations["data_freshness"] = "Data freshness moderate"
            else:
                breakdown.explanations["data_freshness"] = "Data freshness uncertain"
                breakdown.gaps.append("freshness of data unclear")

        return max(0.0, min(10.0, score))

    def _score_query_relevance(
        self,
        findings: Dict[str, Any],
        query: str,
        breakdown: ConfidenceBreakdown
    ) -> float:
        """
        Does the data actually answer what was asked?
        We check for keyword overlap and intent matching.
        """
        score = 0.0
        findings_text = str(findings).lower()
        query_lower = query.lower()

        # filter out the boring words nobody cares about
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "about", "for", "of", "to", "in", "on", "at", "by", "with",
                     "tell", "me", "what", "how", "why", "when", "where", "which",
                     "who", "their", "them", "they", "this", "that", "these", "those"}

        query_words = set(query_lower.split()) - stop_words

        # count how many query words appear in findings
        matches = sum(1 for word in query_words if word in findings_text and len(word) > 2)
        keyword_score = min(4.0, matches * 1.0)
        score += keyword_score

        # did they ask about a specific topic? check if we covered it
        intent_mapping = {
            "stock": ["stock", "trading", "price", "share", "$"],
            "news": ["news", "announced", "launched", "released"],
            "competitor": ["competitor", "rival", "competing", "versus"],
            "financial": ["revenue", "profit", "earnings", "growth", "ytd"],
            "development": ["development", "product", "launched", "released", "new"],
            "ceo": ["ceo", "chief", "executive", "leader"],
        }

        for intent, keywords in intent_mapping.items():
            if any(kw in query_lower for kw in keywords):
                if any(kw in findings_text for kw in keywords):
                    score += 2.0
                    break

        # did we even get the right company?
        company_name = findings.get("company_name", "")
        if company_name:
            company_words = company_name.lower().split()
            if any(word in query_lower for word in company_words):
                score += 2.0

        if keyword_score >= 2:
            breakdown.explanations["query_relevance"] = f"Good keyword match ({matches} terms found)"
        else:
            breakdown.explanations["query_relevance"] = "Limited keyword match"
            breakdown.gaps.append("research may not fully address query")

        return min(10.0, score)

    def _score_data_specificity(
        self,
        findings: Dict[str, Any],
        breakdown: ConfidenceBreakdown
    ) -> float:
        """
        Is this real data or just vague fluff?
        We like seeing numbers, names, specific products, etc.
        """
        score = 0.0
        findings_text = str(findings)

        # look for actual numbers
        number_patterns = [
            r'\$\d+',  # money
            r'\d+%',   # percentages
            r'\d+\s*(billion|million|thousand)',  # big numbers
            r'\d+\.\d+',  # decimals
        ]

        number_count = 0
        for pattern in number_patterns:
            number_count += len(re.findall(pattern, findings_text, re.IGNORECASE))

        score += min(3.0, number_count * 0.5)

        # look for names (crude but works)
        exclude_names = {"the", "a", "an", "inc", "corp", "ltd", "llc", "co"}
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', findings_text)
        named_entities = [w for w in capitalized if w.lower() not in exclude_names]
        unique_entities = len(set(named_entities))

        score += min(3.0, unique_entities * 0.3)

        # mentions of product launches = specific stuff
        product_indicators = ["launched", "released", "introduced", "announced", "unveiled"]
        if any(ind in findings_text.lower() for ind in product_indicators):
            score += 2.0

        # do we even know what company this is about?
        company_name = findings.get("company_name", "")
        if company_name and company_name.lower() not in ["unknown", "unknown company", ""]:
            score += 2.0
        else:
            breakdown.gaps.append("company not specifically identified")

        if score >= 7:
            breakdown.explanations["data_specificity"] = "Highly specific data with concrete facts"
        elif score >= 4:
            breakdown.explanations["data_specificity"] = "Moderately specific data"
        else:
            breakdown.explanations["data_specificity"] = "Data lacks specificity"
            breakdown.gaps.append("data is too generic")

        return min(10.0, score)

    def _score_source_quality(
        self,
        findings: Dict[str, Any],
        breakdown: ConfidenceBreakdown
    ) -> float:
        """
        Where did this data come from? Live API is better than mock data.
        More sources = more confident.
        """
        score = 5.0  # start in the middle

        sources = findings.get("sources", [])
        source_type = findings.get("source", "unknown")

        # more sources = more trust
        if len(sources) >= 3:
            score += 2.0
        elif len(sources) >= 2:
            score += 1.0

        # live api beats mock data
        if "tavily" in source_type.lower() or "api" in source_type.lower():
            score += 2.0
            breakdown.explanations["source_quality"] = "Live API data source"
        elif "mock" in source_type.lower():
            score += 1.0
            breakdown.explanations["source_quality"] = "Mock/cached data source"
        else:
            breakdown.explanations["source_quality"] = "Source type unclear"

        # sanity check: did we get any real data?
        has_news = bool(findings.get("recent_news"))
        has_stock = bool(findings.get("stock_info"))
        has_dev = bool(findings.get("key_developments"))

        fields_present = sum([has_news, has_stock, has_dev])
        if fields_present == 3:
            score += 1.0
        elif fields_present == 0:
            score -= 2.0
            breakdown.gaps.append("no primary data fields populated")

        return max(0.0, min(10.0, score))


# keep one instance around so we don't recreate it every time
_scorer_instance: Optional[ConfidenceScorer] = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Grab the shared scorer instance."""
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = ConfidenceScorer()
    return _scorer_instance


def calculate_hybrid_confidence(
    findings: Dict[str, Any],
    query: str,
    llm_score: Optional[float] = None,
    llm_reasoning: Optional[str] = None
) -> tuple[float, ConfidenceBreakdown]:
    """
    Quick way to score some findings. Returns (score, breakdown).
    """
    scorer = get_confidence_scorer()
    breakdown = scorer.calculate_confidence(findings, query, llm_score, llm_reasoning)
    return breakdown.final_score, breakdown
