"""
Mock company research data for development and testing.

This module provides predefined company data that can be used
when real API calls are not needed or available.
"""

from typing import Dict, Any

# Mock research data for popular companies
MOCK_RESEARCH_DATA: Dict[str, Dict[str, Any]] = {
    "Apple Inc.": {
        "recent_news": "Launched Vision Pro, expanding services revenue. Apple Intelligence features rolling out across iOS 18, macOS Sequoia.",
        "stock_info": "Trading at $195, up 45% YTD. Market cap exceeds $3 trillion. Strong Q4 earnings with services revenue growth.",
        "key_developments": "AI integration across product line with Apple Intelligence. Privacy-focused on-device AI processing. Continued expansion in India manufacturing.",
        "competitors": ["Microsoft", "Google", "Samsung"],
        "industry": "Technology",
        "founded": "1976",
        "ceo": "Tim Cook"
    },
    "Tesla": {
        "recent_news": "Cybertruck deliveries ramping up. Model 3 Highland refresh launched globally. Expanding Supercharger network.",
        "stock_info": "Trading at $242, volatile quarter. Facing increased EV competition. Energy storage business growing rapidly.",
        "key_developments": "FSD v12 rollout with end-to-end neural network. Energy storage growth with Megapack installations. Optimus robot development progressing.",
        "competitors": ["Ford", "GM", "Rivian", "BYD"],
        "industry": "Automotive/Energy",
        "founded": "2003",
        "ceo": "Elon Musk"
    },
    "Microsoft": {
        "recent_news": "Copilot AI integration across Office suite. Azure cloud revenue growth exceeding expectations. GitHub Copilot reaching 1M+ paid subscribers.",
        "stock_info": "Trading at $378, steady growth. Largest company by market cap at times. Strong enterprise demand for AI solutions.",
        "key_developments": "Azure cloud expansion with AI services. OpenAI partnership deepening with GPT-4 integration. Windows 11 Copilot features expanding.",
        "competitors": ["Apple", "Google", "Amazon"],
        "industry": "Technology",
        "founded": "1975",
        "ceo": "Satya Nadella"
    },
    "Google": {
        "recent_news": "Gemini AI model launch competing with GPT-4. YouTube growth continues. Search facing AI disruption concerns.",
        "stock_info": "Trading at $142 (GOOGL), recovering after layoffs. Advertising revenue remains strong. Cloud business achieving profitability.",
        "key_developments": "AI-first strategy with Gemini across products. Cloud market share gains against AWS and Azure. DeepMind breakthroughs in science applications.",
        "competitors": ["Microsoft", "Amazon", "Meta"],
        "industry": "Technology",
        "founded": "1998",
        "ceo": "Sundar Pichai"
    },
    "Amazon": {
        "recent_news": "AWS growth acceleration with AI services. Prime Video expanding with live sports. Logistics automation reducing costs.",
        "stock_info": "Trading at $178, strong e-commerce recovery. AWS remains profit leader. Advertising business growing rapidly.",
        "key_developments": "AI services launch including Bedrock and Q. Logistics automation with robotics. Healthcare expansion with One Medical.",
        "competitors": ["Microsoft", "Walmart", "Alibaba"],
        "industry": "E-commerce/Cloud",
        "founded": "1994",
        "ceo": "Andy Jassy"
    },
    "Meta": {
        "recent_news": "Threads reaching 100M users. Reality Labs losses continuing but VR improving. Instagram Reels competing with TikTok.",
        "stock_info": "Trading at $505, significant recovery from 2022 lows. Ad revenue rebounding. Cost-cutting initiatives showing results.",
        "key_developments": "AI investment across all platforms. Metaverse pivot with Quest 3 launch. Open-source Llama models gaining adoption.",
        "competitors": ["Google", "TikTok", "Snap"],
        "industry": "Social Media/Technology",
        "founded": "2004",
        "ceo": "Mark Zuckerberg"
    },
    "NVIDIA": {
        "recent_news": "AI chip demand exceeding supply. Data center revenue hitting records. Blackwell architecture announced.",
        "stock_info": "Trading at $875, up over 200% in past year. Dominant AI GPU market position. Supply constraints limiting growth.",
        "key_developments": "H100 and H200 chips powering AI revolution. CUDA ecosystem moat strengthening. Automotive and robotics expansion.",
        "competitors": ["AMD", "Intel", "Google TPU"],
        "industry": "Semiconductors",
        "founded": "1993",
        "ceo": "Jensen Huang"
    },
    "Netflix": {
        "recent_news": "Password sharing crackdown boosting subscribers. Ad-supported tier growing. Live events and gaming expansion.",
        "stock_info": "Trading at $485, subscriber growth resuming. Revenue per user improving. International markets driving growth.",
        "key_developments": "Ad-supported tier success. Gaming integration with mobile games. Live events with comedy specials.",
        "competitors": ["Disney+", "Amazon Prime", "HBO Max"],
        "industry": "Entertainment/Streaming",
        "founded": "1997",
        "ceo": "Ted Sarandos & Greg Peters"
    }
}

# Aliases for common variations of company names
COMPANY_ALIASES: Dict[str, str] = {
    # Apple
    "apple": "Apple Inc.",
    "aapl": "Apple Inc.",
    "apple inc": "Apple Inc.",
    "apple computer": "Apple Inc.",

    # Tesla
    "tesla": "Tesla",
    "tsla": "Tesla",
    "tesla inc": "Tesla",
    "tesla motors": "Tesla",

    # Microsoft
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "microsoft corporation": "Microsoft",

    # Google/Alphabet
    "google": "Google",
    "googl": "Google",
    "goog": "Google",
    "alphabet": "Google",
    "alphabet inc": "Google",

    # Amazon
    "amazon": "Amazon",
    "amzn": "Amazon",
    "amazon.com": "Amazon",
    "amazon inc": "Amazon",

    # Meta/Facebook
    "meta": "Meta",
    "facebook": "Meta",
    "fb": "Meta",
    "meta platforms": "Meta",

    # NVIDIA
    "nvidia": "NVIDIA",
    "nvda": "NVIDIA",
    "nvidia corporation": "NVIDIA",

    # Netflix
    "netflix": "Netflix",
    "nflx": "Netflix",
    "netflix inc": "Netflix",
}


def get_company_data(company_name: str) -> Dict[str, Any]:
    """
    Get mock data for a company by name or alias.

    Args:
        company_name: Company name or alias (case-insensitive)

    Returns:
        Dictionary with company research data, or default empty data
    """
    # Normalize the company name
    normalized = company_name.lower().strip()

    # Try to find canonical name via alias
    canonical_name = COMPANY_ALIASES.get(normalized)

    if canonical_name and canonical_name in MOCK_RESEARCH_DATA:
        return MOCK_RESEARCH_DATA[canonical_name]

    # Try direct match (case-insensitive)
    for key in MOCK_RESEARCH_DATA:
        if key.lower() == normalized:
            return MOCK_RESEARCH_DATA[key]

    # Return default data for unknown companies
    return {
        "recent_news": f"No recent news available for {company_name}",
        "stock_info": f"Stock information not available for {company_name}",
        "key_developments": f"No key developments found for {company_name}",
        "note": "This company is not in our mock database. In production, this would use live search."
    }


def list_available_companies() -> list:
    """Return list of companies available in mock data."""
    return list(MOCK_RESEARCH_DATA.keys())
