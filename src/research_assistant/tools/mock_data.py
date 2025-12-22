"""
Mock company research data for development and testing.

This module provides predefined company data that can be used
when real API calls are not needed or available. Includes 25+ companies
across various industries.

Stock prices last updated: December 22, 2024
Data sources: Yahoo Finance, CNBC, Nasdaq, TradingView
"""

from typing import Dict, Any

# Mock research data for popular companies (25+ companies)
# Stock prices updated as of December 20, 2024 (last trading day before Dec 22)
MOCK_RESEARCH_DATA: Dict[str, Dict[str, Any]] = {
    # Big Tech
    "Apple Inc.": {
        "recent_news": "Launched Vision Pro, expanding services revenue. Apple Intelligence features rolling out across iOS 18, macOS Sequoia. Strong holiday quarter expected.",
        "stock_info": "Trading at $249.53, up 34.5% YTD. Market cap exceeds $3.8 trillion. Strong Q4 earnings with services revenue growth. 52-week high of $286.19.",
        "key_developments": "AI integration across product line with Apple Intelligence. Privacy-focused on-device AI processing. Continued expansion in India manufacturing.",
        "competitors": ["Microsoft", "Google", "Samsung"],
        "industry": "Technology",
        "founded": "1976",
        "ceo": "Tim Cook",
        "headquarters": "Cupertino, California",
        "employees": "164,000+",
        "market_cap": "$3.8 trillion"
    },
    "Tesla": {
        "recent_news": "Cybertruck deliveries ramping up. Model 3 Highland refresh launched globally. Stock up 14.6% in December. Strong EV demand despite competition.",
        "stock_info": "Trading at $403.84, significant gains in Q4. Market cap over $1.2 trillion. 52-week high of $489.88. Volatile trading amid EV competition.",
        "key_developments": "FSD v12 rollout with end-to-end neural network. Energy storage growth with Megapack installations. Optimus robot development progressing.",
        "competitors": ["Ford", "GM", "Rivian", "BYD"],
        "industry": "Automotive/Energy",
        "founded": "2003",
        "ceo": "Elon Musk",
        "founder": "Co-founded by Elon Musk, Martin Eberhard, Marc Tarpenning, JB Straubel, Ian Wright",
        "headquarters": "Austin, Texas",
        "employees": "140,000+",
        "market_cap": "$1.3 trillion"
    },
    "Microsoft": {
        "recent_news": "Copilot AI integration across Office suite. Azure cloud revenue growth exceeding expectations. GitHub Copilot reaching 1M+ paid subscribers. Strong enterprise AI demand.",
        "stock_info": "Trading at $419.20, up 13.6% YTD. Among largest companies by market cap at $3.1 trillion. Strong enterprise demand for AI solutions.",
        "key_developments": "Azure cloud expansion with AI services. OpenAI partnership deepening with GPT-4 integration. Windows 11 Copilot features expanding.",
        "competitors": ["Apple", "Google", "Amazon"],
        "industry": "Technology",
        "founded": "1975",
        "ceo": "Satya Nadella",
        "founder": "Bill Gates and Paul Allen",
        "headquarters": "Redmond, Washington",
        "employees": "221,000+",
        "market_cap": "$3.1 trillion"
    },
    "Google": {
        "recent_news": "Gemini 2.0 AI model launch competing with GPT-4. YouTube growth continues. Search facing AI disruption. Stock up 60% over past year.",
        "stock_info": "Trading at $192.50 (GOOGL), strong performance. Advertising revenue remains strong. Cloud business achieving profitability. 52-week range $140.53 - $328.83.",
        "key_developments": "AI-first strategy with Gemini across products. Cloud market share gains against AWS and Azure. DeepMind breakthroughs in science applications.",
        "competitors": ["Microsoft", "Amazon", "Meta"],
        "industry": "Technology",
        "founded": "1998",
        "ceo": "Sundar Pichai",
        "founder": "Larry Page and Sergey Brin",
        "headquarters": "Mountain View, California",
        "employees": "182,000+",
        "market_cap": "$2.4 trillion"
    },
    "Amazon": {
        "recent_news": "AWS growth acceleration with AI services. Prime Video expanding with live sports. Logistics automation reducing costs. Stock trading near all-time highs.",
        "stock_info": "Trading at $227.35, market cap $2.43 trillion. AWS remains profit leader. Advertising business growing rapidly. Up 1.62% over past month.",
        "key_developments": "AI services launch including Bedrock and Q. Logistics automation with robotics. Healthcare expansion with One Medical.",
        "competitors": ["Microsoft", "Walmart", "Alibaba"],
        "industry": "E-commerce/Cloud",
        "founded": "1994",
        "ceo": "Andy Jassy",
        "founder": "Jeff Bezos",
        "headquarters": "Seattle, Washington",
        "employees": "1,500,000+",
        "market_cap": "$2.43 trillion"
    },
    "Meta": {
        "recent_news": "Threads reaching 200M users. Reality Labs losses narrowing. Instagram Reels competing with TikTok. Strong ad revenue recovery.",
        "stock_info": "Trading at $658.77, market cap $1.66 trillion. Significant recovery from 2022 lows. Up 10.96% over past month. 52-week range $479.80 - $796.25.",
        "key_developments": "AI investment across all platforms. Metaverse pivot with Quest 3 launch. Open-source Llama models gaining adoption.",
        "competitors": ["Google", "TikTok", "Snap"],
        "industry": "Social Media/Technology",
        "founded": "2004",
        "ceo": "Mark Zuckerberg",
        "founder": "Mark Zuckerberg",
        "headquarters": "Menlo Park, California",
        "employees": "67,000+",
        "market_cap": "$1.66 trillion"
    },
    "NVIDIA": {
        "recent_news": "AI chip demand exceeding supply. Data center revenue hitting records. Blackwell architecture shipping. Dominant position in AI GPU market.",
        "stock_info": "Trading at $137.49, up significantly YTD. Market cap over $3.3 trillion. 52-week range $86.62 - $212.19. Stock split adjusted.",
        "key_developments": "H100 and H200 chips powering AI revolution. CUDA ecosystem moat strengthening. Automotive and robotics expansion.",
        "competitors": ["AMD", "Intel", "Google TPU"],
        "industry": "Semiconductors",
        "founded": "1993",
        "ceo": "Jensen Huang",
        "founder": "Jensen Huang, Chris Malachowsky, Curtis Priem",
        "headquarters": "Santa Clara, California",
        "employees": "29,000+",
        "market_cap": "$3.3 trillion"
    },
    "Netflix": {
        "recent_news": "Password sharing crackdown boosted subscribers significantly. Ad-supported tier growing. Live events expansion including NFL games on Christmas.",
        "stock_info": "Trading at $909.05, record highs. Market cap $393 billion. Subscriber growth strong. P/E ratio of 46.46. 52-week range $460 - $941.",
        "key_developments": "Ad-supported tier success. Gaming integration with mobile games. Live events with sports and comedy specials.",
        "competitors": ["Disney+", "Amazon Prime", "HBO Max"],
        "industry": "Entertainment/Streaming",
        "founded": "1997",
        "ceo": "Ted Sarandos & Greg Peters",
        "founder": "Reed Hastings and Marc Randolph",
        "headquarters": "Los Gatos, California",
        "employees": "13,000+",
        "market_cap": "$393 billion"
    },
    # Additional Tech Companies
    "AMD": {
        "recent_news": "MI300X AI chips gaining significant market share. Ryzen 9000 series launched. Server CPU gains against Intel continuing.",
        "stock_info": "Trading at $119.21, strong momentum. AI GPU market share growing. Data center revenue accelerating. 52-week range $117 - $227.",
        "key_developments": "MI300X competing with NVIDIA H100. Ryzen AI chips for laptops. EPYC server dominance continuing.",
        "competitors": ["NVIDIA", "Intel", "Qualcomm"],
        "industry": "Semiconductors",
        "founded": "1969",
        "ceo": "Lisa Su",
        "headquarters": "Santa Clara, California",
        "employees": "26,000+",
        "market_cap": "$194 billion"
    },
    "Intel": {
        "recent_news": "Foundry services expansion. Core Ultra mobile chips shipping. Government CHIPS Act funding supporting manufacturing. Leadership transition ongoing.",
        "stock_info": "Trading at $19.66, challenging year. Foundry business investment heavy. PC market recovery helping. 52-week range $18.51 - $51.28.",
        "key_developments": "Intel 4 process node shipping. Gaudi AI accelerators competing. US manufacturing expansion with major investment.",
        "competitors": ["AMD", "NVIDIA", "TSMC"],
        "industry": "Semiconductors",
        "founded": "1968",
        "ceo": "Interim leadership after Pat Gelsinger departure",
        "founder": "Gordon Moore and Robert Noyce",
        "headquarters": "Santa Clara, California",
        "employees": "124,000+",
        "market_cap": "$85 billion"
    },
    "Salesforce": {
        "recent_news": "Agentforce AI platform launch. Einstein GPT AI integration across platform. Strong enterprise demand for AI-powered CRM.",
        "stock_info": "Trading at $347.89, steady growth. Operating margins improving. Market cap $333 billion. Up 28% YTD.",
        "key_developments": "AI-powered CRM with Einstein GPT and Agentforce. Industry cloud solutions expanding. Slack integration deepening.",
        "competitors": ["Microsoft", "Oracle", "SAP"],
        "industry": "Cloud Software/CRM",
        "founded": "1999",
        "ceo": "Marc Benioff",
        "headquarters": "San Francisco, California",
        "employees": "79,000+",
        "market_cap": "$333 billion"
    },
    "Oracle": {
        "recent_news": "Cloud infrastructure growth accelerating. OCI gaining enterprise customers. AI database features launching. Strong cloud momentum.",
        "stock_info": "Trading at $168.70, cloud transition succeeding. Multi-cloud partnerships expanding. Government contracts growing. Market cap $470 billion.",
        "key_developments": "OCI growth competing with AWS/Azure. Cerner healthcare integration. Autonomous database innovations.",
        "competitors": ["Microsoft", "Amazon", "SAP"],
        "industry": "Enterprise Software/Cloud",
        "founded": "1977",
        "ceo": "Safra Catz",
        "founder": "Larry Ellison",
        "headquarters": "Austin, Texas",
        "employees": "164,000+",
        "market_cap": "$470 billion"
    },
    "Adobe": {
        "recent_news": "Firefly AI image generation expanding. Creative Cloud subscriber growth steady. Strong AI-powered feature adoption.",
        "stock_info": "Trading at $438.79, AI features driving growth. Creative Cloud stable. Market cap $195 billion. Digital Experience growth continuing.",
        "key_developments": "Generative AI across Creative suite. Firefly commercial licensing. Document Cloud AI features.",
        "competitors": ["Canva", "Figma", "Microsoft"],
        "industry": "Software/Creative Tools",
        "founded": "1982",
        "ceo": "Shantanu Narayen",
        "headquarters": "San Jose, California",
        "employees": "30,000+",
        "market_cap": "$195 billion"
    },
    # Finance & Fintech
    "JPMorgan Chase": {
        "recent_news": "Record profits in 2024. Investment banking recovery strong. AI trading systems expanding. First Republic acquisition integrated.",
        "stock_info": "Trading at $320.27, up 33% over past year. Largest US bank by assets. P/E ratio 15.71. Dividend yield 1.75%. 52-week range $202.16 - $322.88.",
        "key_developments": "AI fraud detection implementation. Digital banking platform Chase Mobile leading. Commercial banking expansion.",
        "competitors": ["Bank of America", "Wells Fargo", "Citigroup"],
        "industry": "Banking/Finance",
        "founded": "2000 (merger)",
        "ceo": "Jamie Dimon",
        "headquarters": "New York, New York",
        "employees": "310,000+",
        "market_cap": "$872 billion"
    },
    "Visa": {
        "recent_news": "Digital payment volumes reaching records. B2B payments growing. Cross-border transactions strong. Crypto partnerships expanding.",
        "stock_info": "Trading at $317.71, steady dividend growth. Cross-border transactions recovered. New payment flows accelerating. Market cap $680 billion.",
        "key_developments": "Real-time payments expansion. Visa Direct growth. Digital currency partnerships.",
        "competitors": ["Mastercard", "American Express", "PayPal"],
        "industry": "Financial Services/Payments",
        "founded": "1958",
        "ceo": "Ryan McInerney",
        "headquarters": "San Francisco, California",
        "employees": "28,000+",
        "market_cap": "$680 billion"
    },
    "PayPal": {
        "recent_news": "Venmo monetization improving. Checkout experience upgrades launching. New CEO Alex Chriss driving transformation.",
        "stock_info": "Trading at $89.74, recovery in progress. Operating margins focus. Braintree growth continuing. Market cap $95 billion.",
        "key_developments": "PYUSD stablecoin adoption. Checkout optimization. Small business lending growth.",
        "competitors": ["Square", "Stripe", "Apple Pay"],
        "industry": "Fintech/Payments",
        "founded": "1998",
        "ceo": "Alex Chriss",
        "headquarters": "San Jose, California",
        "employees": "27,000+",
        "market_cap": "$95 billion"
    },
    "Square (Block)": {
        "recent_news": "Cash App reaching 57M+ monthly actives. Bitcoin revenue significant. Afterpay integration complete. Strong Q3 earnings.",
        "stock_info": "Trading at $94.70, profitability focus. Bitcoin holdings strategy. Ecosystem integration improving. Market cap $58 billion.",
        "key_developments": "Cash App banking features. Afterpay BNPL integration. TBD Bitcoin platform development.",
        "competitors": ["PayPal", "Stripe", "Shopify"],
        "industry": "Fintech/Payments",
        "founded": "2009",
        "ceo": "Jack Dorsey",
        "founder": "Jack Dorsey and Jim McKelvey",
        "headquarters": "San Francisco, California",
        "employees": "13,000+",
        "market_cap": "$58 billion"
    },
    # Healthcare & Pharma
    "Pfizer": {
        "recent_news": "Post-COVID revenue normalization. Oncology pipeline advancing. Seagen acquisition contributing to growth.",
        "stock_info": "Trading at $25.31, dividend yield attractive at 6.6%. COVID product revenue declined. Pipeline investments increasing. Market cap $143 billion.",
        "key_developments": "mRNA platform expansion beyond COVID. Oncology portfolio growth with Seagen. RSV vaccine Abrysvo approved.",
        "competitors": ["Moderna", "Johnson & Johnson", "Merck"],
        "industry": "Pharmaceuticals",
        "founded": "1849",
        "ceo": "Albert Bourla",
        "headquarters": "New York, New York",
        "employees": "88,000+",
        "market_cap": "$143 billion"
    },
    "Johnson & Johnson": {
        "recent_news": "Kenvue consumer health spinoff complete. MedTech growth continuing. Pharmaceutical pipeline strong with new approvals.",
        "stock_info": "Trading at $144.56, dividend aristocrat. MedTech segment leading growth. Market cap $348 billion. Dividend yield 3.3%.",
        "key_developments": "Kenvue separation completed. Robotics surgery expansion. Oncology and immunology pipelines.",
        "competitors": ["Pfizer", "Abbott", "Medtronic"],
        "industry": "Healthcare/Pharmaceuticals",
        "founded": "1886",
        "ceo": "Joaquin Duato",
        "headquarters": "New Brunswick, New Jersey",
        "employees": "130,000+",
        "market_cap": "$348 billion"
    },
    "UnitedHealth": {
        "recent_news": "Optum health services growing rapidly. Medicare Advantage enrollment strong. Cyberattack recovery complete. Strong fundamentals.",
        "stock_info": "Trading at $510.85, largest health insurer. Optum revenue exceeding insurance. Market cap $475 billion. Value-based care expansion.",
        "key_developments": "AI-powered healthcare analytics. Optum Rx pharmacy growth. Home health services expansion.",
        "competitors": ["CVS Health", "Cigna", "Elevance"],
        "industry": "Healthcare/Insurance",
        "founded": "1977",
        "ceo": "Andrew Witty",
        "headquarters": "Minnetonka, Minnesota",
        "employees": "400,000+",
        "market_cap": "$475 billion"
    },
    # Retail & Consumer
    "Walmart": {
        "recent_news": "E-commerce growth continuing strong. Advertising business Walmart Connect expanding. Automation investments paying off.",
        "stock_info": "Trading at $91.36, largest retailer globally. Grocery market share gains. Walmart+ membership growing. Market cap $735 billion.",
        "key_developments": "Supply chain automation with robotics. Walmart Connect advertising platform. Healthcare clinic expansion.",
        "competitors": ["Amazon", "Target", "Costco"],
        "industry": "Retail",
        "founded": "1962",
        "ceo": "Doug McMillon",
        "founder": "Sam Walton",
        "headquarters": "Bentonville, Arkansas",
        "employees": "2,100,000+",
        "market_cap": "$735 billion"
    },
    "Costco": {
        "recent_news": "Membership fee increase implemented. E-commerce growth accelerating. International expansion continuing. Strong holiday sales.",
        "stock_info": "Trading at $934.18, premium valuation. Same-store sales strong. Membership renewal rates above 93%. Market cap $415 billion.",
        "key_developments": "Digital transformation acceleration. Kirkland Signature brand expansion. Gold bar and silver sales popular.",
        "competitors": ["Walmart", "Target", "BJ's"],
        "industry": "Retail",
        "founded": "1983",
        "ceo": "Ron Vachris",
        "founder": "James Sinegal and Jeffrey Brotman",
        "headquarters": "Issaquah, Washington",
        "employees": "316,000+",
        "market_cap": "$415 billion"
    },
    "Nike": {
        "recent_news": "Direct-to-consumer strategy adjusting. New CEO Elliott Hill taking over. Innovation pipeline launching new products.",
        "stock_info": "Trading at $74.72, inventory normalization. DTC growth moderating. Wholesale partnerships rebuilding. Market cap $111 billion.",
        "key_developments": "Nike App ecosystem expansion. Sustainability initiatives. Jordan brand growth continuing.",
        "competitors": ["Adidas", "Under Armour", "Lululemon"],
        "industry": "Apparel/Footwear",
        "founded": "1964",
        "ceo": "Elliott Hill",
        "founder": "Phil Knight and Bill Bowerman",
        "headquarters": "Beaverton, Oregon",
        "employees": "83,000+",
        "market_cap": "$111 billion"
    },
    "Starbucks": {
        "recent_news": "New CEO Brian Niccol from Chipotle leading turnaround. Back to Starbucks strategy. Mobile ordering improvements.",
        "stock_info": "Trading at $87.60, turnaround in progress. Rewards membership 34M+. Market cap $100 billion. Same-store sales recovering.",
        "key_developments": "Reinvention plan under new leadership. Equipment upgrades for efficiency. Cold beverage innovation.",
        "competitors": ["Dunkin'", "McDonald's", "Local cafes"],
        "industry": "Food & Beverage/Retail",
        "founded": "1971",
        "ceo": "Brian Niccol",
        "founder": "Jerry Baldwin, Zev Siegl, Gordon Bowker",
        "headquarters": "Seattle, Washington",
        "employees": "400,000+",
        "market_cap": "$100 billion"
    },
    # Automotive
    "Toyota": {
        "recent_news": "Hybrid sales surging globally. EV strategy accelerating with new models. Record profits in fiscal year. Production increases.",
        "stock_info": "Trading at $176.83, world's largest automaker by volume. Hybrid leadership strengthening. Market cap $245 billion.",
        "key_developments": "bZ4X and new EV models launching. Hybrid powertrain expansion. Solid-state battery development progress.",
        "competitors": ["Volkswagen", "GM", "Honda", "Tesla"],
        "industry": "Automotive",
        "founded": "1937",
        "ceo": "Koji Sato",
        "founder": "Kiichiro Toyoda",
        "headquarters": "Toyota City, Japan",
        "employees": "375,000+",
        "market_cap": "$245 billion"
    },
    "Ford": {
        "recent_news": "F-150 Lightning production scaling. Mach-E sales growing. Ford Pro commercial business delivering strong profits.",
        "stock_info": "Trading at $9.87, EV investments ongoing. Ford Pro margins high. ICE business funding transition. Market cap $39 billion.",
        "key_developments": "EV platform development. Ford Pro fleet services. BlueCruise hands-free driving updates.",
        "competitors": ["GM", "Tesla", "Toyota", "Rivian"],
        "industry": "Automotive",
        "founded": "1903",
        "ceo": "Jim Farley",
        "founder": "Henry Ford",
        "headquarters": "Dearborn, Michigan",
        "employees": "177,000+",
        "market_cap": "$39 billion"
    },
    # Entertainment
    "Disney": {
        "recent_news": "Streaming profitability achieved. Theme parks strong attendance. ESPN direct-to-consumer launch planned for 2025.",
        "stock_info": "Trading at $112.19, recovery continuing. DTC profitable. Linear TV decline managed. Market cap $203 billion.",
        "key_developments": "Disney+ ad tier success. ESPN streaming platform coming. Parks expansion projects.",
        "competitors": ["Netflix", "Warner Bros", "Universal"],
        "industry": "Entertainment/Media",
        "founded": "1923",
        "ceo": "Bob Iger",
        "founder": "Walt Disney and Roy O. Disney",
        "headquarters": "Burbank, California",
        "employees": "220,000+",
        "market_cap": "$203 billion"
    },
    "Spotify": {
        "recent_news": "Subscriber growth strong at 640M+ monthly actives. Podcast investment rationalizing. Multiple profitable quarters achieved.",
        "stock_info": "Trading at $471.23, profitability achieved. Premium ARPU growing. Audiobook integration. Market cap $95 billion.",
        "key_developments": "AI DJ features. Audiobooks expansion. Creator tools improvement.",
        "competitors": ["Apple Music", "Amazon Music", "YouTube Music"],
        "industry": "Music Streaming",
        "founded": "2006",
        "ceo": "Daniel Ek",
        "founder": "Daniel Ek and Martin Lorentzon",
        "headquarters": "Stockholm, Sweden",
        "employees": "9,000+",
        "market_cap": "$95 billion"
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
    "tesla inc.": "Tesla",
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
