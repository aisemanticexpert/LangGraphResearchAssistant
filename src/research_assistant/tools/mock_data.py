"""
Mock company research data for development and testing.

This module provides predefined company data that can be used
when real API calls are not needed or available. Includes 25+ companies
across various industries.
"""

from typing import Dict, Any

# Mock research data for popular companies (25+ companies)
MOCK_RESEARCH_DATA: Dict[str, Dict[str, Any]] = {
    # Big Tech
    "Apple Inc.": {
        "recent_news": "Launched Vision Pro, expanding services revenue. Apple Intelligence features rolling out across iOS 18, macOS Sequoia.",
        "stock_info": "Trading at $195, up 45% YTD. Market cap exceeds $3 trillion. Strong Q4 earnings with services revenue growth.",
        "key_developments": "AI integration across product line with Apple Intelligence. Privacy-focused on-device AI processing. Continued expansion in India manufacturing.",
        "competitors": ["Microsoft", "Google", "Samsung"],
        "industry": "Technology",
        "founded": "1976",
        "ceo": "Tim Cook",
        "headquarters": "Cupertino, California",
        "employees": "164,000+"
    },
    "Tesla": {
        "recent_news": "Cybertruck deliveries ramping up. Model 3 Highland refresh launched globally. Expanding Supercharger network.",
        "stock_info": "Trading at $242, volatile quarter. Facing increased EV competition. Energy storage business growing rapidly.",
        "key_developments": "FSD v12 rollout with end-to-end neural network. Energy storage growth with Megapack installations. Optimus robot development progressing.",
        "competitors": ["Ford", "GM", "Rivian", "BYD"],
        "industry": "Automotive/Energy",
        "founded": "2003",
        "ceo": "Elon Musk",
        "headquarters": "Austin, Texas",
        "employees": "140,000+"
    },
    "Microsoft": {
        "recent_news": "Copilot AI integration across Office suite. Azure cloud revenue growth exceeding expectations. GitHub Copilot reaching 1M+ paid subscribers.",
        "stock_info": "Trading at $378, steady growth. Largest company by market cap at times. Strong enterprise demand for AI solutions.",
        "key_developments": "Azure cloud expansion with AI services. OpenAI partnership deepening with GPT-4 integration. Windows 11 Copilot features expanding.",
        "competitors": ["Apple", "Google", "Amazon"],
        "industry": "Technology",
        "founded": "1975",
        "ceo": "Satya Nadella",
        "headquarters": "Redmond, Washington",
        "employees": "221,000+"
    },
    "Google": {
        "recent_news": "Gemini AI model launch competing with GPT-4. YouTube growth continues. Search facing AI disruption concerns.",
        "stock_info": "Trading at $142 (GOOGL), recovering after layoffs. Advertising revenue remains strong. Cloud business achieving profitability.",
        "key_developments": "AI-first strategy with Gemini across products. Cloud market share gains against AWS and Azure. DeepMind breakthroughs in science applications.",
        "competitors": ["Microsoft", "Amazon", "Meta"],
        "industry": "Technology",
        "founded": "1998",
        "ceo": "Sundar Pichai",
        "headquarters": "Mountain View, California",
        "employees": "182,000+"
    },
    "Amazon": {
        "recent_news": "AWS growth acceleration with AI services. Prime Video expanding with live sports. Logistics automation reducing costs.",
        "stock_info": "Trading at $178, strong e-commerce recovery. AWS remains profit leader. Advertising business growing rapidly.",
        "key_developments": "AI services launch including Bedrock and Q. Logistics automation with robotics. Healthcare expansion with One Medical.",
        "competitors": ["Microsoft", "Walmart", "Alibaba"],
        "industry": "E-commerce/Cloud",
        "founded": "1994",
        "ceo": "Andy Jassy",
        "headquarters": "Seattle, Washington",
        "employees": "1,500,000+"
    },
    "Meta": {
        "recent_news": "Threads reaching 100M users. Reality Labs losses continuing but VR improving. Instagram Reels competing with TikTok.",
        "stock_info": "Trading at $505, significant recovery from 2022 lows. Ad revenue rebounding. Cost-cutting initiatives showing results.",
        "key_developments": "AI investment across all platforms. Metaverse pivot with Quest 3 launch. Open-source Llama models gaining adoption.",
        "competitors": ["Google", "TikTok", "Snap"],
        "industry": "Social Media/Technology",
        "founded": "2004",
        "ceo": "Mark Zuckerberg",
        "headquarters": "Menlo Park, California",
        "employees": "67,000+"
    },
    "NVIDIA": {
        "recent_news": "AI chip demand exceeding supply. Data center revenue hitting records. Blackwell architecture announced.",
        "stock_info": "Trading at $875, up over 200% in past year. Dominant AI GPU market position. Supply constraints limiting growth.",
        "key_developments": "H100 and H200 chips powering AI revolution. CUDA ecosystem moat strengthening. Automotive and robotics expansion.",
        "competitors": ["AMD", "Intel", "Google TPU"],
        "industry": "Semiconductors",
        "founded": "1993",
        "ceo": "Jensen Huang",
        "headquarters": "Santa Clara, California",
        "employees": "29,000+"
    },
    "Netflix": {
        "recent_news": "Password sharing crackdown boosting subscribers. Ad-supported tier growing. Live events and gaming expansion.",
        "stock_info": "Trading at $485, subscriber growth resuming. Revenue per user improving. International markets driving growth.",
        "key_developments": "Ad-supported tier success. Gaming integration with mobile games. Live events with comedy specials.",
        "competitors": ["Disney+", "Amazon Prime", "HBO Max"],
        "industry": "Entertainment/Streaming",
        "founded": "1997",
        "ceo": "Ted Sarandos & Greg Peters",
        "headquarters": "Los Gatos, California",
        "employees": "13,000+"
    },
    # Additional Tech Companies
    "AMD": {
        "recent_news": "MI300X AI chips gaining market share. Ryzen 9000 series launching. Server CPU gains against Intel continuing.",
        "stock_info": "Trading at $165, strong momentum. AI GPU market share growing. Data center revenue accelerating.",
        "key_developments": "MI300X competing with NVIDIA H100. Ryzen AI chips for laptops. EPYC server dominance continuing.",
        "competitors": ["NVIDIA", "Intel", "Qualcomm"],
        "industry": "Semiconductors",
        "founded": "1969",
        "ceo": "Lisa Su",
        "headquarters": "Santa Clara, California",
        "employees": "26,000+"
    },
    "Intel": {
        "recent_news": "Foundry services expansion with TSMC partnership. Meteor Lake laptop chips shipping. Government CHIPS Act funding received.",
        "stock_info": "Trading at $45, turnaround in progress. Foundry business investment heavy. PC market recovery helping.",
        "key_developments": "Intel 4 process node shipping. Gaudi AI accelerators competing. US manufacturing expansion with $20B investment.",
        "competitors": ["AMD", "NVIDIA", "TSMC"],
        "industry": "Semiconductors",
        "founded": "1968",
        "ceo": "Pat Gelsinger",
        "headquarters": "Santa Clara, California",
        "employees": "124,000+"
    },
    "Salesforce": {
        "recent_news": "Einstein GPT AI integration across platform. Slack growth accelerating. Data Cloud gaining enterprise adoption.",
        "stock_info": "Trading at $265, activist investor pressure resolved. Operating margins improving. M&A strategy evolving.",
        "key_developments": "AI-powered CRM with Einstein GPT. Industry cloud solutions expanding. Slack integration deepening.",
        "competitors": ["Microsoft", "Oracle", "SAP"],
        "industry": "Cloud Software/CRM",
        "founded": "1999",
        "ceo": "Marc Benioff",
        "headquarters": "San Francisco, California",
        "employees": "79,000+"
    },
    "Oracle": {
        "recent_news": "Cloud infrastructure growth accelerating. Healthcare acquisitions integrating. AI database features launching.",
        "stock_info": "Trading at $125, cloud transition succeeding. Multi-cloud partnerships expanding. Government contracts growing.",
        "key_developments": "OCI growth competing with AWS/Azure. Cerner healthcare integration. Autonomous database innovations.",
        "competitors": ["Microsoft", "Amazon", "SAP"],
        "industry": "Enterprise Software/Cloud",
        "founded": "1977",
        "ceo": "Safra Catz",
        "headquarters": "Austin, Texas",
        "employees": "164,000+"
    },
    "Adobe": {
        "recent_news": "Firefly AI image generation expanding. Creative Cloud subscriber growth steady. Figma acquisition blocked by regulators.",
        "stock_info": "Trading at $575, AI features driving growth. Creative Cloud stable. Digital Experience growth continuing.",
        "key_developments": "Generative AI across Creative suite. Firefly commercial licensing. Document Cloud AI features.",
        "competitors": ["Canva", "Figma", "Microsoft"],
        "industry": "Software/Creative Tools",
        "founded": "1982",
        "ceo": "Shantanu Narayen",
        "headquarters": "San Jose, California",
        "employees": "30,000+"
    },
    # Finance & Fintech
    "JPMorgan Chase": {
        "recent_news": "First Republic acquisition completed. Investment banking recovery beginning. AI trading systems expanding.",
        "stock_info": "Trading at $172, largest US bank by assets. Net interest income strong. Credit card growth continuing.",
        "key_developments": "AI fraud detection implementation. Digital banking platform Chase Mobile leading. Commercial banking expansion.",
        "competitors": ["Bank of America", "Wells Fargo", "Citigroup"],
        "industry": "Banking/Finance",
        "founded": "2000 (merger)",
        "ceo": "Jamie Dimon",
        "headquarters": "New York, New York",
        "employees": "310,000+"
    },
    "Visa": {
        "recent_news": "Digital payment volumes reaching records. B2B payments growing. Crypto partnerships expanding.",
        "stock_info": "Trading at $275, steady dividend growth. Cross-border transactions recovering. New payment flows accelerating.",
        "key_developments": "Real-time payments expansion. Visa Direct growth. Digital currency partnerships.",
        "competitors": ["Mastercard", "American Express", "PayPal"],
        "industry": "Financial Services/Payments",
        "founded": "1958",
        "ceo": "Ryan McInerney",
        "headquarters": "San Francisco, California",
        "employees": "28,000+"
    },
    "PayPal": {
        "recent_news": "Venmo monetization improving. Checkout experience upgrades launching. Crypto trading features expanding.",
        "stock_info": "Trading at $62, activist pressure ongoing. Operating margins focus. Braintree growth continuing.",
        "key_developments": "PYUSD stablecoin launch. Checkout optimization. Small business lending growth.",
        "competitors": ["Square", "Stripe", "Apple Pay"],
        "industry": "Fintech/Payments",
        "founded": "1998",
        "ceo": "Alex Chriss",
        "headquarters": "San Jose, California",
        "employees": "27,000+"
    },
    "Square (Block)": {
        "recent_news": "Cash App reaching 50M+ monthly actives. Bitcoin revenue significant. Afterpay integration complete.",
        "stock_info": "Trading at $68, profitability focus. Bitcoin holdings strategy. Ecosystem integration improving.",
        "key_developments": "Cash App banking features. Afterpay BNPL integration. TBD Bitcoin platform development.",
        "competitors": ["PayPal", "Stripe", "Shopify"],
        "industry": "Fintech/Payments",
        "founded": "2009",
        "ceo": "Jack Dorsey",
        "headquarters": "San Francisco, California",
        "employees": "13,000+"
    },
    # Healthcare & Pharma
    "Pfizer": {
        "recent_news": "Post-COVID revenue normalization. Oncology pipeline advancing. Seagen acquisition completed.",
        "stock_info": "Trading at $28, dividend yield attractive. COVID product revenue declining. Pipeline investments increasing.",
        "key_developments": "mRNA platform expansion beyond COVID. Oncology portfolio growth with Seagen. RSV vaccine approval.",
        "competitors": ["Moderna", "Johnson & Johnson", "Merck"],
        "industry": "Pharmaceuticals",
        "founded": "1849",
        "ceo": "Albert Bourla",
        "headquarters": "New York, New York",
        "employees": "88,000+"
    },
    "Johnson & Johnson": {
        "recent_news": "Kenvue consumer health spinoff complete. MedTech growth continuing. Pharmaceutical pipeline strong.",
        "stock_info": "Trading at $158, dividend aristocrat. MedTech segment leading growth. Talc litigation reserves taken.",
        "key_developments": "Kenvue separation completed. Robotics surgery expansion. Oncology and immunology pipelines.",
        "competitors": ["Pfizer", "Abbott", "Medtronic"],
        "industry": "Healthcare/Pharmaceuticals",
        "founded": "1886",
        "ceo": "Joaquin Duato",
        "headquarters": "New Brunswick, New Jersey",
        "employees": "130,000+"
    },
    "UnitedHealth": {
        "recent_news": "Optum health services growing rapidly. Medicare Advantage enrollment strong. AI healthcare tools deploying.",
        "stock_info": "Trading at $525, largest health insurer. Optum revenue exceeding insurance. Value-based care expansion.",
        "key_developments": "AI-powered healthcare analytics. Optum Rx pharmacy growth. Home health services expansion.",
        "competitors": ["CVS Health", "Cigna", "Anthem"],
        "industry": "Healthcare/Insurance",
        "founded": "1977",
        "ceo": "Andrew Witty",
        "headquarters": "Minnetonka, Minnesota",
        "employees": "400,000+"
    },
    # Retail & Consumer
    "Walmart": {
        "recent_news": "E-commerce growth continuing. Advertising business expanding. Automation investments paying off.",
        "stock_info": "Trading at $165, largest retailer globally. Grocery market share gains. Walmart+ membership growing.",
        "key_developments": "Supply chain automation with robotics. Walmart Connect advertising platform. Healthcare clinic expansion.",
        "competitors": ["Amazon", "Target", "Costco"],
        "industry": "Retail",
        "founded": "1962",
        "ceo": "Doug McMillon",
        "headquarters": "Bentonville, Arkansas",
        "employees": "2,100,000+"
    },
    "Costco": {
        "recent_news": "Membership fee increase announced. E-commerce growth accelerating. International expansion continuing.",
        "stock_info": "Trading at $705, premium valuation. Same-store sales strong. Membership renewal rates above 90%.",
        "key_developments": "Digital transformation acceleration. Kirkland Signature brand expansion. Gold bar sales popular.",
        "competitors": ["Walmart", "Target", "BJ's"],
        "industry": "Retail",
        "founded": "1983",
        "ceo": "Ron Vachris",
        "headquarters": "Issaquah, Washington",
        "employees": "316,000+"
    },
    "Nike": {
        "recent_news": "Direct-to-consumer strategy adjusting. China recovery uncertain. Innovation pipeline launching.",
        "stock_info": "Trading at $98, inventory normalization. DTC growth moderating. Wholesale partnerships rebuilding.",
        "key_developments": "Nike App ecosystem expansion. Sustainability initiatives. Jordan brand growth continuing.",
        "competitors": ["Adidas", "Under Armour", "Lululemon"],
        "industry": "Apparel/Footwear",
        "founded": "1964",
        "ceo": "John Donahoe",
        "headquarters": "Beaverton, Oregon",
        "employees": "83,000+"
    },
    "Starbucks": {
        "recent_news": "New CEO Laxman Narasimhan leading turnaround. China reopening boost. Mobile ordering dominant.",
        "stock_info": "Trading at $95, same-store sales growth. Rewards membership 32M+. International expansion.",
        "key_developments": "Reinvention plan execution. Equipment upgrades for efficiency. Cold beverage innovation.",
        "competitors": ["Dunkin'", "McDonald's", "Local cafes"],
        "industry": "Food & Beverage/Retail",
        "founded": "1971",
        "ceo": "Laxman Narasimhan",
        "headquarters": "Seattle, Washington",
        "employees": "400,000+"
    },
    # Automotive
    "Toyota": {
        "recent_news": "Hybrid sales surging. EV strategy accelerating. Record profits reported.",
        "stock_info": "Trading at $245, world's largest automaker by volume. Hybrid leadership strengthening. Solid-state battery development.",
        "key_developments": "bZ4X EV improvements. Hybrid powertrain expansion. Woven City smart city project.",
        "competitors": ["Volkswagen", "GM", "Honda", "Tesla"],
        "industry": "Automotive",
        "founded": "1937",
        "ceo": "Koji Sato",
        "headquarters": "Toyota City, Japan",
        "employees": "375,000+"
    },
    "Ford": {
        "recent_news": "F-150 Lightning production scaling. Mach-E sales growing. Ford Pro commercial business strong.",
        "stock_info": "Trading at $12, EV investments heavy. Ford Pro margins high. ICE business funding transition.",
        "key_developments": "EV platform development. Ford Pro fleet services. BlueCruise hands-free driving.",
        "competitors": ["GM", "Tesla", "Toyota", "Rivian"],
        "industry": "Automotive",
        "founded": "1903",
        "ceo": "Jim Farley",
        "headquarters": "Dearborn, Michigan",
        "employees": "177,000+"
    },
    # Entertainment
    "Disney": {
        "recent_news": "Streaming losses narrowing. Theme parks record attendance. ESPN strategy evolving.",
        "stock_info": "Trading at $92, activist investor settled. DTC path to profitability. Linear TV decline managed.",
        "key_developments": "Disney+ ad tier launch. ESPN streaming platform coming. Parks expansion in Asia.",
        "competitors": ["Netflix", "Warner Bros", "Universal"],
        "industry": "Entertainment/Media",
        "founded": "1923",
        "ceo": "Bob Iger",
        "headquarters": "Burbank, California",
        "employees": "220,000+"
    },
    "Spotify": {
        "recent_news": "Subscriber growth strong. Podcast investment rationalizing. Price increases implemented.",
        "stock_info": "Trading at $195, first profitable quarter. Premium ARPU growing. Audiobook integration.",
        "key_developments": "AI DJ features. Audiobooks expansion. Creator tools improvement.",
        "competitors": ["Apple Music", "Amazon Music", "YouTube Music"],
        "industry": "Music Streaming",
        "founded": "2006",
        "ceo": "Daniel Ek",
        "headquarters": "Stockholm, Sweden",
        "employees": "9,000+"
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

    # AMD
    "amd": "AMD",
    "advanced micro devices": "AMD",

    # Intel
    "intel": "Intel",
    "intc": "Intel",
    "intel corporation": "Intel",

    # Salesforce
    "salesforce": "Salesforce",
    "crm": "Salesforce",
    "salesforce.com": "Salesforce",

    # Oracle
    "oracle": "Oracle",
    "orcl": "Oracle",
    "oracle corporation": "Oracle",

    # Adobe
    "adobe": "Adobe",
    "adbe": "Adobe",
    "adobe inc": "Adobe",
    "adobe systems": "Adobe",

    # JPMorgan
    "jpmorgan": "JPMorgan Chase",
    "jpm": "JPMorgan Chase",
    "jpmorgan chase": "JPMorgan Chase",
    "chase": "JPMorgan Chase",
    "jp morgan": "JPMorgan Chase",

    # Visa
    "visa": "Visa",
    "v": "Visa",
    "visa inc": "Visa",

    # PayPal
    "paypal": "PayPal",
    "pypl": "PayPal",
    "paypal holdings": "PayPal",

    # Square/Block
    "square": "Square (Block)",
    "block": "Square (Block)",
    "sq": "Square (Block)",
    "block inc": "Square (Block)",

    # Pfizer
    "pfizer": "Pfizer",
    "pfe": "Pfizer",
    "pfizer inc": "Pfizer",

    # Johnson & Johnson
    "johnson & johnson": "Johnson & Johnson",
    "j&j": "Johnson & Johnson",
    "jnj": "Johnson & Johnson",
    "johnson and johnson": "Johnson & Johnson",

    # UnitedHealth
    "unitedhealth": "UnitedHealth",
    "unh": "UnitedHealth",
    "united health": "UnitedHealth",
    "unitedhealth group": "UnitedHealth",

    # Walmart
    "walmart": "Walmart",
    "wmt": "Walmart",
    "wal-mart": "Walmart",

    # Costco
    "costco": "Costco",
    "cost": "Costco",
    "costco wholesale": "Costco",

    # Nike
    "nike": "Nike",
    "nke": "Nike",
    "nike inc": "Nike",

    # Starbucks
    "starbucks": "Starbucks",
    "sbux": "Starbucks",
    "starbucks corporation": "Starbucks",

    # Toyota
    "toyota": "Toyota",
    "tm": "Toyota",
    "toyota motor": "Toyota",

    # Ford
    "ford": "Ford",
    "f": "Ford",
    "ford motor": "Ford",
    "ford motors": "Ford",

    # Disney
    "disney": "Disney",
    "dis": "Disney",
    "walt disney": "Disney",
    "the walt disney company": "Disney",

    # Spotify
    "spotify": "Spotify",
    "spot": "Spotify",
    "spotify technology": "Spotify",
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
