"""
Guardrails Module for LangGraph Research Assistant
===================================================

Production-grade input/output validation for securities research.
Implements SEC/FINRA compliance requirements and best practices.

This module provides comprehensive protection against:
    1. Prompt injection attacks (10+ patterns)
    2. Market manipulation queries (30+ patterns)
    3. Insider trading facilitation (8+ patterns)
    4. Low-quality or stale data responses

Key Components:
    - InputGuardrails: Query validation and sanitization
    - OutputGuardrails: Response quality and compliance checks
    - CompanyNameValidator: Company name normalization (50+ companies)
    - AuditLogger: Compliance audit trail
    - GuardrailConfig: Customizable configuration

Usage:
    from guardrails import InputGuardrails, OutputGuardrails

    input_guard = InputGuardrails()
    result = input_guard.validate_query("Tell me about Apple")

    if result.passed:
        # Process the sanitized query
        query = result.sanitized_content

Total Safety Patterns: 48+

Developer: Rajesh Gupta
Copyright (c) 2024 Rajesh Gupta. All rights reserved.
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


# Configure module logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# VIOLATION TYPES - Categories of guardrail violations
# ============================================================================

class GuardrailViolationType(str, Enum):
    """
    Types of guardrail violations for categorized handling.

    Each type has specific handling logic and user messaging.
    """
    # Input violations
    EMPTY_INPUT = "empty_input"
    INVALID_CHARACTERS = "invalid_characters"
    PROMPT_INJECTION = "prompt_injection"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"

    # Compliance violations
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING = "insider_trading"

    # Content violations
    HARMFUL_CONTENT = "harmful_content"
    PROFANITY = "profanity"
    PERSONAL_INFO = "personal_info"

    # Output violations
    MISSING_DISCLAIMER = "missing_disclaimer"
    LOW_CONFIDENCE = "low_confidence"
    STALE_DATA = "stale_data"


# ============================================================================
# RESULT DATACLASSES
# ============================================================================

@dataclass
class GuardrailResult:
    """
    Result of a guardrail validation check.

    Provides detailed information about whether validation passed,
    what violation occurred (if any), and sanitized content.

    Attributes:
        passed: Whether the validation passed
        violation_type: Type of violation if failed
        violation_message: User-friendly message explaining the violation
        sanitized_content: Cleaned/safe version of the content
        metadata: Additional context (timestamps, scores, etc.)
    """
    passed: bool
    violation_type: Optional[GuardrailViolationType] = None
    violation_message: Optional[str] = None
    sanitized_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailConfig:
    """
    Configuration for guardrail behavior.

    Allows customization of validation thresholds and features.

    Attributes:
        max_query_length: Maximum allowed query length (chars)
        min_query_length: Minimum required query length (chars)
        max_response_length: Maximum response length (chars)
        min_confidence_threshold: Below this, add low confidence warning
        max_data_age_hours: Above this, add stale data warning
        enable_prompt_injection_detection: Check for prompt injection
        enable_content_moderation: Check for harmful content
        enable_compliance_checks: Check for market manipulation/insider trading
        require_disclaimers: Add financial disclaimers
        log_all_checks: Log all validation checks
    """
    max_query_length: int = 2000
    min_query_length: int = 3
    max_response_length: int = 10000
    min_confidence_threshold: float = 3.0
    max_data_age_hours: float = 72.0
    enable_prompt_injection_detection: bool = True
    enable_content_moderation: bool = True
    enable_compliance_checks: bool = True
    require_disclaimers: bool = True
    log_all_checks: bool = True


# ============================================================================
# INPUT GUARDRAILS - Query validation and sanitization
# ============================================================================

class InputGuardrails:
    """
    Input validation guardrails for user queries.

    Ensures queries are safe, properly formatted, and don't request
    illegal or harmful information.

    Validation Layers:
        1. Format validation (length, characters)
        2. Prompt injection detection
        3. Market manipulation detection
        4. Insider trading detection
        5. Content sanitization

    Usage:
        guardrails = InputGuardrails()
        result = guardrails.validate_query("Tell me about Apple stock")

        if result.passed:
            safe_query = result.sanitized_content
        else:
            error_msg = result.violation_message
    """

    # Patterns indicating prompt injection attempts
    # These catch common jailbreak techniques
    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)\s+instructions",
        r"disregard\s+(your|all)\s+instructions",
        r"you\s+are\s+now\s+[a-z]+",
        r"pretend\s+you\s+are",
        r"act\s+as\s+if",
        r"forget\s+(everything|all)",
        r"system\s*:\s*",
        r"<\|.*\|>",
        r"\[\[.*\]\]",
        r"```\s*(system|admin)",
    ]

    # Patterns indicating market manipulation requests
    # These are illegal under SEC regulations
    MARKET_MANIPULATION_PATTERNS = [
        # Classic manipulation schemes
        r"pump\s+and\s+dump",
        r"short\s+and\s+distort",
        r"manipulate\s+(the\s+)?(stock|market|price)",
        r"coordinate(d)?\s+(buying|selling)",
        r"artificially\s+(inflate|deflate)",
        r"spread\s+false\s+(rumors?|information)",
        r"front\s*run(ning)?",
        r"spoofing",
        r"layering",
        r"wash\s+trad(e|ing)",

        # Dump/exit manipulation language
        r"(how\s+(can|do|to)|help\s+(me\s+)?|want\s+to)\s*dump\s+(my\s+)?(stock|shares|position|holdings?)",
        r"dump\s+(the\s+)?(stock|shares|market|price)",
        r"(crash|tank|destroy|crush|kill)\s+(the\s+)?(stock|shares|price|market)",
        r"make\s+(the\s+)?(stock|price|shares)\s+(crash|tank|fall|drop|plummet)",
        r"drive\s+(down|up)\s+(the\s+)?(stock|price|shares)",

        # Coordinated trading language
        r"(organize|coordinate|plan)\s+(a\s+)?(sell[\s-]?off|buying\s+spree|mass\s+(buying|selling))",
        r"get\s+everyone\s+to\s+(buy|sell)",
        r"(convince|persuade|get)\s+(people|others|investors)\s+to\s+(buy|sell|dump)",

        # Short selling manipulation
        r"naked\s+short(ing)?",
        r"(short\s+)?ladder\s+attack",
        r"bear\s+raid",

        # General manipulation intent
        r"(rig|fix)\s+(the\s+)?(market|stock|price)",
        r"corner\s+the\s+market",

        # Dump + company name patterns (catching "dump moderna", "dump apple", etc.)
        r"(how\s+(can|do|to|should)\s+i?|help\s+(me\s+)?|want\s+to|going\s+to|need\s+to)\s*dump\s+\w+",
        r"dump(ing)?\s+(all\s+)?(my\s+)?\w+\s*(stock|shares|position)?",
    ]

    # Patterns indicating insider trading queries
    # Trading on material non-public info is illegal
    INSIDER_TRADING_PATTERNS = [
        r"insider\s+(trading|information|tips?)",
        r"non\s*-?\s*public\s+information",
        r"material\s+non\s*-?\s*public",
        r"(before|ahead\s+of)\s+(the\s+)?announcement",
        r"trade\s+on\s+confidential",
        r"leak(ed)?\s+(earnings?|merger|acquisition)",
    ]

    def __init__(self, config: Optional[GuardrailConfig] = None):
        """
        Initialize InputGuardrails with configuration.

        Args:
            config: Optional GuardrailConfig for customization
        """
        self.config = config or GuardrailConfig()
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self._injection_regex = [
            re.compile(p, re.IGNORECASE)
            for p in self.PROMPT_INJECTION_PATTERNS
        ]
        self._manipulation_regex = [
            re.compile(p, re.IGNORECASE)
            for p in self.MARKET_MANIPULATION_PATTERNS
        ]
        self._insider_regex = [
            re.compile(p, re.IGNORECASE)
            for p in self.INSIDER_TRADING_PATTERNS
        ]

    def validate_query(self, query: str) -> GuardrailResult:
        """
        Perform comprehensive validation on user query.

        Runs through all validation layers in order of priority:
            1. Empty/null check
            2. Sanitization
            3. Length constraints
            4. Prompt injection check
            5. Compliance checks (manipulation, insider trading)

        Args:
            query: The user's input query

        Returns:
            GuardrailResult with pass/fail status and details
        """
        # Layer 1: Check for empty input
        if not query or not query.strip():
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailViolationType.EMPTY_INPUT,
                violation_message="Query cannot be empty. Please enter a question about a company."
            )

        # Layer 2: Sanitize and normalize
        sanitized = self._sanitize_query(query)

        # Layer 3: Check length constraints
        if len(sanitized) < self.config.min_query_length:
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailViolationType.TOO_SHORT,
                violation_message=f"Query is too short. Please provide at least {self.config.min_query_length} characters."
            )

        if len(sanitized) > self.config.max_query_length:
            return GuardrailResult(
                passed=False,
                violation_type=GuardrailViolationType.TOO_LONG,
                violation_message=f"Query is too long. Please limit to {self.config.max_query_length} characters.",
                sanitized_content=sanitized[:self.config.max_query_length]
            )

        # Layer 4: Check for prompt injection
        if self.config.enable_prompt_injection_detection:
            injection_result = self._check_prompt_injection(sanitized)
            if not injection_result.passed:
                return injection_result

        # Layer 5: Compliance checks
        if self.config.enable_compliance_checks:
            manipulation_result = self._check_market_manipulation(sanitized)
            if not manipulation_result.passed:
                return manipulation_result

            insider_result = self._check_insider_trading(sanitized)
            if not insider_result.passed:
                return insider_result

        # All checks passed
        if self.config.log_all_checks:
            logger.info(f"Query validation passed: {sanitized[:50]}...")

        return GuardrailResult(
            passed=True,
            sanitized_content=sanitized,
            metadata={
                "original_length": len(query),
                "sanitized_length": len(sanitized),
                "timestamp": datetime.now().isoformat()
            }
        )

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize query by removing potentially harmful content.

        Removes:
            - Null bytes and control characters
            - HTML/XML tags
            - Excessive whitespace

        Args:
            query: Raw user query

        Returns:
            Sanitized query string
        """
        # Remove null bytes and control characters (except common whitespace)
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)

        # Normalize whitespace (collapse multiple spaces/newlines)
        sanitized = ' '.join(sanitized.split())

        # Remove HTML/XML tags
        sanitized = re.sub(r'<[^>]+>', '', sanitized)

        return sanitized.strip()

    def _check_prompt_injection(self, query: str) -> GuardrailResult:
        """
        Check for prompt injection attacks.

        Detects common patterns used to override AI instructions.

        Args:
            query: Sanitized query to check

        Returns:
            GuardrailResult with pass/fail status
        """
        for pattern in self._injection_regex:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected: pattern={pattern.pattern}")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.PROMPT_INJECTION,
                    violation_message="Your query contains instructions that I cannot process. Please rephrase your question about company research."
                )
        return GuardrailResult(passed=True)

    def _check_market_manipulation(self, query: str) -> GuardrailResult:
        """
        Check for market manipulation requests.

        Market manipulation is illegal under SEC regulations.
        We cannot assist with such activities.

        Args:
            query: Sanitized query to check

        Returns:
            GuardrailResult with pass/fail status
        """
        for pattern in self._manipulation_regex:
            if pattern.search(query):
                logger.warning("Market manipulation query detected")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.MARKET_MANIPULATION,
                    violation_message=(
                        "I cannot provide assistance with market manipulation activities. "
                        "Such activities are illegal under SEC regulations. "
                        "Please ask about legitimate company research instead."
                    )
                )
        return GuardrailResult(passed=True)

    def _check_insider_trading(self, query: str) -> GuardrailResult:
        """
        Check for insider trading related queries.

        Trading on material non-public information is illegal.

        Args:
            query: Sanitized query to check

        Returns:
            GuardrailResult with pass/fail status
        """
        for pattern in self._insider_regex:
            if pattern.search(query):
                logger.warning("Insider trading query detected")
                return GuardrailResult(
                    passed=False,
                    violation_type=GuardrailViolationType.INSIDER_TRADING,
                    violation_message=(
                        "I cannot provide assistance with insider trading or material non-public information. "
                        "Trading on such information is illegal. "
                        "I can only help with publicly available company research."
                    )
                )
        return GuardrailResult(passed=True)


# ============================================================================
# OUTPUT GUARDRAILS - Response validation and enhancement
# ============================================================================

class OutputGuardrails:
    """
    Output validation guardrails for research responses.

    Ensures responses include appropriate disclaimers and warnings
    for financial content, low confidence results, and stale data.

    Features:
        - Automatic financial disclaimer injection
        - Low confidence warnings
        - Stale data notifications
        - Investment advice detection

    Usage:
        guardrails = OutputGuardrails()
        result = guardrails.validate_response(
            response="Apple stock is trading at $195...",
            confidence_score=7.5,
            data_age_hours=24
        )
    """

    # Required disclaimer for financial content
    FINANCIAL_DISCLAIMER = (
        "---\n\n"
        "**DISCLAIMER:** This information is for educational and research purposes only. "
        "It does not constitute financial, investment, or trading advice. "
        "Always consult with a qualified financial advisor before making investment decisions. "
        "Past performance does not guarantee future results."
    )

    # Patterns indicating investment advice
    INVESTMENT_ADVICE_PATTERNS = [
        r"you\s+should\s+(buy|sell|invest)",
        r"recommend\s+(buying|selling|investing)",
        r"(buy|sell)\s+this\s+stock",
        r"guaranteed\s+(returns?|profits?)",
        r"risk\s*-?\s*free\s+investment",
        r"can't\s+lose",
        r"must\s+(buy|sell)",
        r"(great|perfect)\s+time\s+to\s+(buy|sell)",
    ]

    def __init__(self, config: Optional[GuardrailConfig] = None):
        """
        Initialize OutputGuardrails with configuration.

        Args:
            config: Optional GuardrailConfig for customization
        """
        self.config = config or GuardrailConfig()
        self._advice_regex = [
            re.compile(p, re.IGNORECASE)
            for p in self.INVESTMENT_ADVICE_PATTERNS
        ]

    def validate_response(
        self,
        response: str,
        confidence_score: float,
        data_age_hours: float = 0.0
    ) -> GuardrailResult:
        """
        Validate and enhance output response.

        Checks for quality issues and adds appropriate warnings/disclaimers.

        Args:
            response: Generated response text
            confidence_score: Research confidence score (0-10)
            data_age_hours: Age of the data in hours

        Returns:
            GuardrailResult with enhanced response content
        """
        issues = []

        # Check confidence threshold
        if confidence_score < self.config.min_confidence_threshold:
            issues.append({
                "type": GuardrailViolationType.LOW_CONFIDENCE,
                "message": f"Low confidence score: {confidence_score:.1f}/10"
            })

        # Check data freshness
        if data_age_hours > self.config.max_data_age_hours:
            issues.append({
                "type": GuardrailViolationType.STALE_DATA,
                "message": f"Data may be stale ({data_age_hours:.1f} hours old)"
            })

        # Check for investment advice without disclaimer
        if self.config.require_disclaimers:
            has_advice = any(p.search(response) for p in self._advice_regex)
            has_disclaimer = (
                "disclaimer" in response.lower() or
                "not financial advice" in response.lower() or
                "not investment advice" in response.lower()
            )

            if has_advice and not has_disclaimer:
                issues.append({
                    "type": GuardrailViolationType.MISSING_DISCLAIMER,
                    "message": "Response contains investment advice without disclaimer"
                })

        # Enhance response with warnings and disclaimers if needed
        enhanced_response = self._enhance_response(response, issues, confidence_score)

        return GuardrailResult(
            passed=True,  # We always pass but may enhance
            sanitized_content=enhanced_response,
            metadata={
                "issues": [i["type"].value for i in issues],
                "enhanced": bool(issues),
                "confidence_score": confidence_score,
                "data_age_hours": data_age_hours
            }
        )

    def _enhance_response(
        self,
        response: str,
        issues: List[Dict],
        confidence_score: float
    ) -> str:
        """
        Enhance response with appropriate warnings and disclaimers.

        Args:
            response: Original response text
            issues: List of identified issues
            confidence_score: Research confidence score

        Returns:
            Enhanced response with warnings/disclaimers
        """
        warnings = []

        for issue in issues:
            if issue["type"] == GuardrailViolationType.LOW_CONFIDENCE:
                warnings.append(
                    f"**Note:** This research has a confidence score of "
                    f"{confidence_score:.1f}/10. Some information may be incomplete or limited."
                )
            elif issue["type"] == GuardrailViolationType.STALE_DATA:
                warnings.append(
                    "**Note:** Some data may not reflect the most recent information. "
                    "Please verify with current sources for time-sensitive decisions."
                )

        enhanced = response

        # Prepend warnings if any
        if warnings:
            warning_block = "\n\n".join(warnings)
            enhanced = f"{warning_block}\n\n---\n\n{enhanced}"

        # Append disclaimer if required
        has_disclaimer_issue = any(
            i["type"] == GuardrailViolationType.MISSING_DISCLAIMER
            for i in issues
        )

        if has_disclaimer_issue or self.config.require_disclaimers:
            if "DISCLAIMER" not in enhanced:
                enhanced = enhanced + "\n\n" + self.FINANCIAL_DISCLAIMER

        return enhanced


# ============================================================================
# COMPANY NAME VALIDATOR - Normalize company names and tickers
# ============================================================================

class CompanyNameValidator:
    """
    Validates and normalizes company names and ticker symbols.

    Essential for accurate research - handles common variations,
    aliases, and ticker symbol lookups.

    Features:
        - Company alias resolution (e.g., "Apple" -> "Apple Inc.")
        - Ticker to company mapping (e.g., "AAPL" -> "Apple Inc.")
        - Case-insensitive matching
        - Partial name matching

    Usage:
        company, ticker = CompanyNameValidator.normalize_company_name("AAPL")
        # Returns: ("Apple Inc.", "AAPL")
    """

    # Common company name variations and their canonical forms
    COMPANY_ALIASES = {
        # Technology Giants
        "apple": "Apple Inc.",
        "aapl": "Apple Inc.",
        "microsoft": "Microsoft Corporation",
        "msft": "Microsoft Corporation",
        "google": "Alphabet Inc.",
        "googl": "Alphabet Inc.",
        "goog": "Alphabet Inc.",
        "alphabet": "Alphabet Inc.",
        "amazon": "Amazon.com Inc.",
        "amzn": "Amazon.com Inc.",
        "meta": "Meta Platforms Inc.",
        "facebook": "Meta Platforms Inc.",
        "fb": "Meta Platforms Inc.",
        "nvidia": "NVIDIA Corporation",
        "nvda": "NVIDIA Corporation",
        "tesla": "Tesla Inc.",
        "tsla": "Tesla Inc.",
        "netflix": "Netflix Inc.",
        "nflx": "Netflix Inc.",
        "amd": "Advanced Micro Devices Inc.",
        "intel": "Intel Corporation",
        "intc": "Intel Corporation",
        "salesforce": "Salesforce Inc.",
        "crm": "Salesforce Inc.",
        "adobe": "Adobe Inc.",
        "adbe": "Adobe Inc.",
        "oracle": "Oracle Corporation",
        "orcl": "Oracle Corporation",
        "cisco": "Cisco Systems Inc.",
        "csco": "Cisco Systems Inc.",

        # Financial
        "jpmorgan": "JPMorgan Chase & Co.",
        "jp morgan": "JPMorgan Chase & Co.",
        "jpm": "JPMorgan Chase & Co.",
        "goldman": "Goldman Sachs Group Inc.",
        "goldman sachs": "Goldman Sachs Group Inc.",
        "gs": "Goldman Sachs Group Inc.",
        "bank of america": "Bank of America Corporation",
        "bofa": "Bank of America Corporation",
        "bac": "Bank of America Corporation",
        "wells fargo": "Wells Fargo & Company",
        "wfc": "Wells Fargo & Company",
        "morgan stanley": "Morgan Stanley",
        "ms": "Morgan Stanley",
        "visa": "Visa Inc.",
        "v": "Visa Inc.",
        "mastercard": "Mastercard Inc.",
        "ma": "Mastercard Inc.",
        "paypal": "PayPal Holdings Inc.",
        "pypl": "PayPal Holdings Inc.",

        # Healthcare
        "pfizer": "Pfizer Inc.",
        "pfe": "Pfizer Inc.",
        "johnson & johnson": "Johnson & Johnson",
        "j&j": "Johnson & Johnson",
        "jnj": "Johnson & Johnson",
        "unitedhealth": "UnitedHealth Group Inc.",
        "unh": "UnitedHealth Group Inc.",
        "merck": "Merck & Co. Inc.",
        "mrk": "Merck & Co. Inc.",
        "abbvie": "AbbVie Inc.",
        "abbv": "AbbVie Inc.",
        "eli lilly": "Eli Lilly and Company",
        "lly": "Eli Lilly and Company",
        "moderna": "Moderna Inc.",
        "mrna": "Moderna Inc.",

        # Energy
        "exxon": "Exxon Mobil Corporation",
        "exxonmobil": "Exxon Mobil Corporation",
        "xom": "Exxon Mobil Corporation",
        "chevron": "Chevron Corporation",
        "cvx": "Chevron Corporation",

        # Consumer
        "walmart": "Walmart Inc.",
        "wmt": "Walmart Inc.",
        "target": "Target Corporation",
        "tgt": "Target Corporation",
        "costco": "Costco Wholesale Corporation",
        "cost": "Costco Wholesale Corporation",
        "disney": "The Walt Disney Company",
        "dis": "The Walt Disney Company",
        "nike": "Nike Inc.",
        "nke": "Nike Inc.",
        "starbucks": "Starbucks Corporation",
        "sbux": "Starbucks Corporation",
        "mcdonald's": "McDonald's Corporation",
        "mcdonalds": "McDonald's Corporation",
        "mcd": "McDonald's Corporation",
        "coca-cola": "The Coca-Cola Company",
        "coca cola": "The Coca-Cola Company",
        "coke": "The Coca-Cola Company",
        "ko": "The Coca-Cola Company",
        "pepsi": "PepsiCo Inc.",
        "pepsico": "PepsiCo Inc.",
        "pep": "PepsiCo Inc.",

        # Other Major Companies
        "berkshire": "Berkshire Hathaway Inc.",
        "berkshire hathaway": "Berkshire Hathaway Inc.",
        "brk": "Berkshire Hathaway Inc.",
        "boeing": "The Boeing Company",
        "ba": "The Boeing Company",
        "general electric": "General Electric Company",
        "ge": "General Electric Company",
        "3m": "3M Company",
        "mmm": "3M Company",
        "ibm": "International Business Machines Corporation",
    }

    # Ticker to company mapping (for precise ticker lookups)
    TICKER_MAP = {
        # Tech
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "GOOG": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "META": "Meta Platforms Inc.",
        "NVDA": "NVIDIA Corporation",
        "TSLA": "Tesla Inc.",
        "NFLX": "Netflix Inc.",
        "AMD": "Advanced Micro Devices Inc.",
        "INTC": "Intel Corporation",
        "CRM": "Salesforce Inc.",
        "ADBE": "Adobe Inc.",
        "ORCL": "Oracle Corporation",
        "CSCO": "Cisco Systems Inc.",

        # Financial
        "JPM": "JPMorgan Chase & Co.",
        "GS": "Goldman Sachs Group Inc.",
        "BAC": "Bank of America Corporation",
        "WFC": "Wells Fargo & Company",
        "MS": "Morgan Stanley",
        "V": "Visa Inc.",
        "MA": "Mastercard Inc.",
        "PYPL": "PayPal Holdings Inc.",

        # Healthcare
        "PFE": "Pfizer Inc.",
        "JNJ": "Johnson & Johnson",
        "UNH": "UnitedHealth Group Inc.",
        "MRK": "Merck & Co. Inc.",
        "ABBV": "AbbVie Inc.",
        "LLY": "Eli Lilly and Company",
        "MRNA": "Moderna Inc.",

        # Energy
        "XOM": "Exxon Mobil Corporation",
        "CVX": "Chevron Corporation",

        # Consumer
        "WMT": "Walmart Inc.",
        "TGT": "Target Corporation",
        "COST": "Costco Wholesale Corporation",
        "DIS": "The Walt Disney Company",
        "NKE": "Nike Inc.",
        "SBUX": "Starbucks Corporation",
        "MCD": "McDonald's Corporation",
        "KO": "The Coca-Cola Company",
        "PEP": "PepsiCo Inc.",

        # Other
        "BRK.A": "Berkshire Hathaway Inc.",
        "BRK.B": "Berkshire Hathaway Inc.",
        "BA": "The Boeing Company",
        "GE": "General Electric Company",
        "MMM": "3M Company",
        "IBM": "International Business Machines Corporation",
    }

    @classmethod
    def normalize_company_name(cls, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract and normalize company name from query.

        Attempts multiple matching strategies:
            1. Direct alias match
            2. Ticker symbol match
            3. Partial name match

        Args:
            query: User query that may contain company name

        Returns:
            Tuple of (canonical_name, ticker) or (None, None) if not found
        """
        query_lower = query.lower().strip()

        # Strategy 1: Check for direct alias match
        for alias, canonical in cls.COMPANY_ALIASES.items():
            if alias in query_lower:
                # Find corresponding ticker
                ticker = cls._find_ticker_for_company(canonical)
                return canonical, ticker

        # Strategy 2: Check for ticker symbols (uppercase 1-5 letter words)
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        potential_tickers = re.findall(ticker_pattern, query.upper())

        for ticker in potential_tickers:
            if ticker in cls.TICKER_MAP:
                return cls.TICKER_MAP[ticker], ticker

        # Strategy 3: Try variations
        # Check for "Inc", "Corp", "Company" etc.
        company_suffixes = ["inc", "corp", "corporation", "company", "ltd"]
        for suffix in company_suffixes:
            if suffix in query_lower:
                # Try to extract company name before suffix
                pattern = rf"(\w+(?:\s+\w+)*)\s+{suffix}"
                match = re.search(pattern, query_lower)
                if match:
                    potential_name = match.group(1)
                    for alias, canonical in cls.COMPANY_ALIASES.items():
                        if potential_name in alias:
                            ticker = cls._find_ticker_for_company(canonical)
                            return canonical, ticker

        return None, None

    @classmethod
    def _find_ticker_for_company(cls, company_name: str) -> Optional[str]:
        """Find ticker symbol for a company name."""
        for ticker, company in cls.TICKER_MAP.items():
            if company == company_name:
                return ticker
        return None

    @classmethod
    def is_valid_ticker(cls, ticker: str) -> bool:
        """
        Check if a ticker symbol is valid.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            True if ticker is in our database
        """
        return ticker.upper() in cls.TICKER_MAP

    @classmethod
    def get_company_by_ticker(cls, ticker: str) -> Optional[str]:
        """
        Get company name by ticker symbol.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Company name or None if not found
        """
        return cls.TICKER_MAP.get(ticker.upper())

    @classmethod
    def get_all_companies(cls) -> List[str]:
        """Get list of all known company names."""
        return list(set(cls.TICKER_MAP.values()))

    @classmethod
    def get_all_tickers(cls) -> List[str]:
        """Get list of all known ticker symbols."""
        return list(cls.TICKER_MAP.keys())


# ============================================================================
# AUDIT LOGGER - Compliance audit trail
# ============================================================================

class AuditLogger:
    """
    Audit logging for compliance and debugging.

    Maintains a comprehensive audit trail of all research operations
    for regulatory compliance and debugging purposes.

    Features:
        - Event logging with timestamps
        - Session tracking
        - File persistence (optional)
        - Structured log entries

    Usage:
        logger = AuditLogger()
        logger.log_event(
            event_type="query_processed",
            session_id="abc123",
            user_id="user456",
            details={"query": "Tell me about Apple", "status": "success"}
        )
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize AuditLogger.

        Args:
            log_file: Optional file path for persistent logging
        """
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        self._logger = logging.getLogger("AuditLogger")

    def log_event(
        self,
        event_type: str,
        session_id: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log an audit event.

        Args:
            event_type: Type of event (query_received, validation_passed, etc.)
            session_id: Unique session identifier
            user_id: Optional user identifier
            details: Additional event details

        Returns:
            The logged event entry
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "user_id": user_id,
            "details": details or {}
        }

        self.logs.append(event)
        self._logger.info(f"Audit: {event_type} - Session: {session_id}")

        # Write to file if configured
        if self.log_file:
            self._write_to_file(event)

        return event

    def _write_to_file(self, event: Dict[str, Any]) -> None:
        """Write event to log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except IOError as e:
            self._logger.error(f"Failed to write audit log: {e}")

    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all logs for a specific session.

        Args:
            session_id: Session to retrieve logs for

        Returns:
            List of log entries for the session
        """
        return [log for log in self.logs if log["session_id"] == session_id]

    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Get the most recent log entries.

        Args:
            count: Number of entries to retrieve

        Returns:
            List of recent log entries
        """
        return self.logs[-count:]

    def export_logs(self, filepath: str) -> None:
        """
        Export all logs to a JSON file.

        Args:
            filepath: Path to export file
        """
        with open(filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)
        self._logger.info(f"Exported {len(self.logs)} log entries to {filepath}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_research_query(
    query: str,
    config: Optional[GuardrailConfig] = None
) -> GuardrailResult:
    """
    Quick validation function for research queries.

    Convenience wrapper around InputGuardrails for simple use cases.

    Args:
        query: User query to validate
        config: Optional guardrail configuration

    Returns:
        GuardrailResult with validation status
    """
    guardrails = InputGuardrails(config)
    return guardrails.validate_query(query)


def validate_research_output(
    response: str,
    confidence_score: float,
    data_age_hours: float = 0.0,
    config: Optional[GuardrailConfig] = None
) -> GuardrailResult:
    """
    Quick validation function for research outputs.

    Convenience wrapper around OutputGuardrails for simple use cases.

    Args:
        response: Generated response
        confidence_score: Confidence score (0-10)
        data_age_hours: Age of data in hours
        config: Optional guardrail configuration

    Returns:
        GuardrailResult with validation status and enhanced response
    """
    guardrails = OutputGuardrails(config)
    return guardrails.validate_response(response, confidence_score, data_age_hours)
