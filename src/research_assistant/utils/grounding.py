"""
Grounding check - catch hallucinations before they go out.

When the LLM writes a response, we want to make sure it's actually
based on the research data and not making stuff up. This module
checks numbers, dates, names, and quotes against the source.

Not perfect (regex-based), but catches the obvious problems.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """What we found when checking if the response is backed by data."""

    is_grounded: bool = True
    grounding_score: float = 1.0
    grounded_claims: List[str] = field(default_factory=list)  # stuff we verified
    ungrounded_claims: List[str] = field(default_factory=list)  # stuff we couldn't find
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "grounding_score": round(self.grounding_score, 2),
            "grounded_claims": self.grounded_claims,
            "ungrounded_claims": self.ungrounded_claims,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class ResponseGroundingValidator:
    """
    Checks if the LLM response is actually backed by the source data.

    We look for numbers, dates, names, and quotes in the response,
    then check if they appear in the research data. If not, flag them.
    """

    # regex for finding specific things in text
    NUMBER_PATTERN = re.compile(r'\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand))?|\d+(?:\.\d+)?%|\d+(?:,\d{3})+')
    DATE_PATTERN = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b|\bQ[1-4]\s*\d{4}\b|\b20\d{2}\b')
    QUOTE_PATTERN = re.compile(r'"([^"]+)"')

    # words that usually come before a factual claim
    CLAIM_INDICATORS = [
        "announced", "reported", "stated", "confirmed", "revealed",
        "launched", "released", "introduced", "achieved", "reached"
    ]

    def __init__(self):
        self._claim_indicator_pattern = re.compile(
            r'\b(' + '|'.join(self.CLAIM_INDICATORS) + r')\b',
            re.IGNORECASE
        )

    def validate(
        self,
        response: str,
        source_data: Dict[str, Any],
        strict: bool = False
    ) -> GroundingResult:
        """
        Check if the response is backed by the source data.

        strict=True means any unverified claim = fail.
        strict=False allows some wiggle room (70%+ must be grounded).
        """
        result = GroundingResult()

        # flatten source data into one big string for searching
        source_text = self._normalize_source_data(source_data)
        source_text_lower = source_text.lower()

        # check each type of claim
        self._validate_numbers(response, source_text, result)
        self._validate_dates(response, source_text, result)
        self._validate_names(response, source_data, result)
        self._validate_quotes(response, source_text, result)
        self._validate_factual_claims(response, source_text_lower, result)

        # what percentage of claims could we verify?
        total_claims = len(result.grounded_claims) + len(result.ungrounded_claims)
        if total_claims > 0:
            result.grounding_score = len(result.grounded_claims) / total_claims
        else:
            result.grounding_score = 1.0  # nothing to check = assume ok

        # decide pass/fail
        if strict:
            result.is_grounded = len(result.ungrounded_claims) == 0
        else:
            result.is_grounded = result.grounding_score >= 0.7

        # helpful suggestions
        if not result.is_grounded:
            result.recommendations.append(
                "Review and remove or verify ungrounded claims before presenting response"
            )
        if result.warnings:
            result.recommendations.append(
                "Consider rephrasing claims that couldn't be verified"
            )

        logger.debug(f"Grounding validation result: {result.to_dict()}")

        return result

    def _normalize_source_data(self, source_data: Dict[str, Any]) -> str:
        """Flatten the source dict into one big searchable string."""
        parts = []

        def extract_text(obj, prefix=""):
            if isinstance(obj, str):
                parts.append(obj)
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    extract_text(value, f"{prefix}{key}: ")
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item, prefix)

        extract_text(source_data)
        return " ".join(parts)

    def _validate_numbers(
        self,
        response: str,
        source_text: str,
        result: GroundingResult
    ) -> None:
        """Make sure numbers in the response actually came from somewhere."""
        response_numbers = set(self.NUMBER_PATTERN.findall(response))
        source_numbers = set(self.NUMBER_PATTERN.findall(source_text))

        for number in response_numbers:
            normalized_num = number.replace(",", "").strip()

            found = False
            for source_num in source_numbers:
                source_normalized = source_num.replace(",", "").strip()
                if normalized_num == source_normalized:
                    found = True
                    break
                # try numerical comparison too (handles $10 vs $10.00)
                try:
                    if abs(float(normalized_num.replace("$", "").replace("%", "")) -
                           float(source_normalized.replace("$", "").replace("%", ""))) < 0.01:
                        found = True
                        break
                except ValueError:
                    pass

            if found:
                result.grounded_claims.append(f"Number: {number}")
            else:
                result.ungrounded_claims.append(f"Number: {number}")
                result.warnings.append(f"Number '{number}' not found in source data")

    def _validate_dates(
        self,
        response: str,
        source_text: str,
        result: GroundingResult
    ) -> None:
        """Check that dates mentioned are from the source."""
        response_dates = set(self.DATE_PATTERN.findall(response))
        source_dates = set(self.DATE_PATTERN.findall(source_text))

        for date in response_dates:
            if date in source_dates or date.lower() in source_text.lower():
                result.grounded_claims.append(f"Date: {date}")
            else:
                # at least check if the year is right
                year_match = re.search(r'20\d{2}', date)
                if year_match and year_match.group() in source_text:
                    result.grounded_claims.append(f"Date: {date} (year verified)")
                    result.warnings.append(f"Full date '{date}' not verified, but year found")
                else:
                    result.ungrounded_claims.append(f"Date: {date}")
                    result.warnings.append(f"Date '{date}' not found in source data")

    def _validate_names(
        self,
        response: str,
        source_data: Dict[str, Any],
        result: GroundingResult
    ) -> None:
        """Check that people mentioned are actually in the data."""
        source_text = self._normalize_source_data(source_data)

        # look for CEO/executive names specifically
        ceo_pattern = re.compile(r'(?:CEO|chief executive|led by|headed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE)
        ceo_matches = ceo_pattern.findall(response)

        for name in ceo_matches:
            if name in source_text:
                result.grounded_claims.append(f"Person: {name}")
            else:
                # also check the ceo field specifically
                ceo_in_source = source_data.get("ceo") or \
                               source_data.get("additional_info", {}).get("ceo") or \
                               source_data.get("raw_data", {}).get("additional_info", {}).get("ceo")

                if ceo_in_source and name.lower() in ceo_in_source.lower():
                    result.grounded_claims.append(f"Person: {name}")
                else:
                    result.ungrounded_claims.append(f"Person: {name}")
                    result.warnings.append(f"Name '{name}' not verified in source data")

    def _validate_quotes(
        self,
        response: str,
        source_text: str,
        result: GroundingResult
    ) -> None:
        """If there's a quote, it better be from the source."""
        quotes = self.QUOTE_PATTERN.findall(response)

        for quote in quotes:
            if len(quote) > 10:  # skip tiny quotes
                if quote.lower() in source_text.lower():
                    result.grounded_claims.append(f"Quote: \"{quote[:30]}...\"")
                else:
                    # might be paraphrased - warn but don't fail
                    result.warnings.append(f"Quote not found verbatim: \"{quote[:30]}...\"")

    def _validate_factual_claims(
        self,
        response: str,
        source_text_lower: str,
        result: GroundingResult
    ) -> None:
        """Check sentences with 'announced', 'reported', etc."""
        sentences = re.split(r'[.!?]', response)

        for sentence in sentences:
            if self._claim_indicator_pattern.search(sentence):
                # grab the important words
                claim_words = set(re.findall(r'\b[a-z]{4,}\b', sentence.lower()))
                claim_words -= {'that', 'this', 'have', 'been', 'with', 'from', 'they', 'their'}

                # if most words are in source, probably legit
                words_found = sum(1 for w in claim_words if w in source_text_lower)
                if claim_words and words_found / len(claim_words) >= 0.5:
                    result.grounded_claims.append(f"Claim: {sentence[:50].strip()}...")
                elif claim_words:
                    result.warnings.append(f"Claim may be embellished: {sentence[:50].strip()}...")


# keep one instance around
_validator_instance: Optional[ResponseGroundingValidator] = None


def get_grounding_validator() -> ResponseGroundingValidator:
    """Grab the shared validator."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ResponseGroundingValidator()
    return _validator_instance


def validate_response_grounding(
    response: str,
    source_data: Dict[str, Any],
    strict: bool = False
) -> GroundingResult:
    """Quick way to check if a response is grounded in the data."""
    validator = get_grounding_validator()
    return validator.validate(response, source_data, strict)
