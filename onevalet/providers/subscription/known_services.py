"""Known subscription services — sender domain to service metadata mapping.

Used for fast rule-based detection before LLM extraction.
When a sender domain matches, we already know the service_name and category,
and only need LLM to extract amount/cycle from the email content.
"""

from typing import Dict

ServiceInfo = Dict[str, str]  # {"name": ..., "category": ...}

KNOWN_SERVICES: Dict[str, ServiceInfo] = {
    # ── Streaming ──
    "netflix.com": {"name": "Netflix", "category": "streaming"},
    "spotify.com": {"name": "Spotify", "category": "streaming"},
    "hulu.com": {"name": "Hulu", "category": "streaming"},
    "disneyplus.com": {"name": "Disney+", "category": "streaming"},
    "hbomax.com": {"name": "Max", "category": "streaming"},
    "max.com": {"name": "Max", "category": "streaming"},
    "crunchyroll.com": {"name": "Crunchyroll", "category": "streaming"},
    "paramountplus.com": {"name": "Paramount+", "category": "streaming"},
    "peacocktv.com": {"name": "Peacock", "category": "streaming"},
    "appletv.com": {"name": "Apple TV+", "category": "streaming"},
    "tidal.com": {"name": "Tidal", "category": "streaming"},
    "pandora.com": {"name": "Pandora", "category": "streaming"},
    "deezer.com": {"name": "Deezer", "category": "streaming"},
    "audible.com": {"name": "Audible", "category": "streaming"},
    "twitch.tv": {"name": "Twitch", "category": "streaming"},
    "primevideo.com": {"name": "Prime Video", "category": "streaming"},
    "youtube.com": {"name": "YouTube Premium", "category": "streaming"},
    "sling.com": {"name": "Sling TV", "category": "streaming"},
    "fubo.tv": {"name": "FuboTV", "category": "streaming"},

    # ── Cloud / Storage ──
    "apple.com": {"name": "Apple", "category": "cloud"},
    "icloud.com": {"name": "iCloud", "category": "cloud"},
    "google.com": {"name": "Google One", "category": "cloud"},
    "dropbox.com": {"name": "Dropbox", "category": "cloud"},
    "onedrive.com": {"name": "OneDrive", "category": "cloud"},
    "box.com": {"name": "Box", "category": "cloud"},
    "pcloud.com": {"name": "pCloud", "category": "cloud"},

    # ── Productivity / Office ──
    "microsoft.com": {"name": "Microsoft 365", "category": "productivity"},
    "office.com": {"name": "Microsoft 365", "category": "productivity"},
    "evernote.com": {"name": "Evernote", "category": "productivity"},
    "notion.so": {"name": "Notion", "category": "productivity"},
    "todoist.com": {"name": "Todoist", "category": "productivity"},
    "trello.com": {"name": "Trello", "category": "productivity"},
    "asana.com": {"name": "Asana", "category": "productivity"},
    "monday.com": {"name": "Monday.com", "category": "productivity"},
    "clickup.com": {"name": "ClickUp", "category": "productivity"},
    "airtable.com": {"name": "Airtable", "category": "productivity"},
    "miro.com": {"name": "Miro", "category": "productivity"},
    "slack.com": {"name": "Slack", "category": "productivity"},
    "zoom.us": {"name": "Zoom", "category": "productivity"},
    "zoom.com": {"name": "Zoom", "category": "productivity"},

    # ── SaaS / Tools ──
    "adobe.com": {"name": "Adobe Creative Cloud", "category": "saas"},
    "figma.com": {"name": "Figma", "category": "saas"},
    "canva.com": {"name": "Canva", "category": "saas"},
    "openai.com": {"name": "ChatGPT Plus", "category": "saas"},
    "chatgpt.com": {"name": "ChatGPT Plus", "category": "saas"},
    "grammarly.com": {"name": "Grammarly", "category": "saas"},
    "1password.com": {"name": "1Password", "category": "saas"},
    "lastpass.com": {"name": "LastPass", "category": "saas"},
    "bitwarden.com": {"name": "Bitwarden", "category": "saas"},
    "dashlane.com": {"name": "Dashlane", "category": "saas"},

    # ── Developer Tools ──
    "github.com": {"name": "GitHub", "category": "developer"},
    "gitlab.com": {"name": "GitLab", "category": "developer"},
    "atlassian.com": {"name": "Atlassian", "category": "developer"},
    "vercel.com": {"name": "Vercel", "category": "developer"},
    "heroku.com": {"name": "Heroku", "category": "developer"},
    "digitalocean.com": {"name": "DigitalOcean", "category": "developer"},
    "cloudflare.com": {"name": "Cloudflare", "category": "developer"},
    "netlify.com": {"name": "Netlify", "category": "developer"},
    "render.com": {"name": "Render", "category": "developer"},
    "railway.app": {"name": "Railway", "category": "developer"},
    "supabase.com": {"name": "Supabase", "category": "developer"},
    "mongodb.com": {"name": "MongoDB Atlas", "category": "developer"},

    # ── Telecom / Internet ──
    "t-mobile.com": {"name": "T-Mobile", "category": "telecom"},
    "verizon.com": {"name": "Verizon", "category": "telecom"},
    "att.com": {"name": "AT&T", "category": "telecom"},
    "xfinity.com": {"name": "Xfinity", "category": "telecom"},
    "comcast.com": {"name": "Xfinity", "category": "telecom"},
    "sprint.com": {"name": "Sprint", "category": "telecom"},
    "mintmobile.com": {"name": "Mint Mobile", "category": "telecom"},
    "visible.com": {"name": "Visible", "category": "telecom"},
    "cricketwireless.com": {"name": "Cricket", "category": "telecom"},
    "boostmobile.com": {"name": "Boost Mobile", "category": "telecom"},
    "fi.google.com": {"name": "Google Fi", "category": "telecom"},
    "spectrum.com": {"name": "Spectrum", "category": "telecom"},
    "cox.com": {"name": "Cox", "category": "telecom"},
    "frontier.com": {"name": "Frontier", "category": "telecom"},
    "centurylink.com": {"name": "CenturyLink", "category": "telecom"},

    # ── VPN / Security ──
    "nordvpn.com": {"name": "NordVPN", "category": "vpn"},
    "expressvpn.com": {"name": "ExpressVPN", "category": "vpn"},
    "surfshark.com": {"name": "Surfshark", "category": "vpn"},
    "proton.me": {"name": "Proton", "category": "vpn"},
    "protonmail.com": {"name": "Proton", "category": "vpn"},
    "tunnelbear.com": {"name": "TunnelBear", "category": "vpn"},
    "privateinternetaccess.com": {"name": "PIA", "category": "vpn"},
    "nordpass.com": {"name": "NordPass", "category": "vpn"},

    # ── Fitness / Health ──
    "onepeloton.com": {"name": "Peloton", "category": "fitness"},
    "strava.com": {"name": "Strava", "category": "fitness"},
    "headspace.com": {"name": "Headspace", "category": "fitness"},
    "calm.com": {"name": "Calm", "category": "fitness"},
    "myfitnesspal.com": {"name": "MyFitnessPal", "category": "fitness"},
    "whoop.com": {"name": "WHOOP", "category": "fitness"},
    "noom.com": {"name": "Noom", "category": "fitness"},
    "fitbit.com": {"name": "Fitbit Premium", "category": "fitness"},

    # ── News / Media ──
    "nytimes.com": {"name": "New York Times", "category": "news"},
    "washingtonpost.com": {"name": "Washington Post", "category": "news"},
    "wsj.com": {"name": "Wall Street Journal", "category": "news"},
    "medium.com": {"name": "Medium", "category": "news"},
    "substack.com": {"name": "Substack", "category": "news"},
    "economist.com": {"name": "The Economist", "category": "news"},
    "theathletic.com": {"name": "The Athletic", "category": "news"},

    # ── Gaming ──
    "playstation.com": {"name": "PlayStation Plus", "category": "gaming"},
    "xbox.com": {"name": "Xbox Game Pass", "category": "gaming"},
    "nintendo.com": {"name": "Nintendo Online", "category": "gaming"},
    "steampowered.com": {"name": "Steam", "category": "gaming"},
    "ea.com": {"name": "EA Play", "category": "gaming"},
    "epicgames.com": {"name": "Epic Games", "category": "gaming"},
    "riotgames.com": {"name": "Riot Games", "category": "gaming"},

    # ── Education ──
    "coursera.org": {"name": "Coursera", "category": "education"},
    "udemy.com": {"name": "Udemy", "category": "education"},
    "skillshare.com": {"name": "Skillshare", "category": "education"},
    "masterclass.com": {"name": "MasterClass", "category": "education"},
    "brilliant.org": {"name": "Brilliant", "category": "education"},
    "duolingo.com": {"name": "Duolingo Plus", "category": "education"},
    "linkedin.com": {"name": "LinkedIn Premium", "category": "education"},

    # ── Finance / Investing ──
    "robinhood.com": {"name": "Robinhood Gold", "category": "finance"},
    "coinbase.com": {"name": "Coinbase One", "category": "finance"},
    "tradingview.com": {"name": "TradingView", "category": "finance"},

    # ── Home / Smart Home ──
    "ring.com": {"name": "Ring", "category": "home"},
    "nest.com": {"name": "Nest Aware", "category": "home"},
    "simplisafe.com": {"name": "SimpliSafe", "category": "home"},
    "adt.com": {"name": "ADT", "category": "home"},
    "vivint.com": {"name": "Vivint", "category": "home"},

    # ── Shopping / Memberships ──
    "amazon.com": {"name": "Amazon Prime", "category": "shopping"},
    "costco.com": {"name": "Costco", "category": "shopping"},
    "walmart.com": {"name": "Walmart+", "category": "shopping"},
    "instacart.com": {"name": "Instacart+", "category": "shopping"},
    "doordash.com": {"name": "DashPass", "category": "shopping"},
    "ubereats.com": {"name": "Uber One", "category": "shopping"},
    "grubhub.com": {"name": "Grubhub+", "category": "shopping"},
}

# Keywords that suggest an email is subscription-related (case-insensitive).
# Used when sender domain is NOT in KNOWN_SERVICES.
SUBSCRIPTION_KEYWORDS = [
    "receipt", "invoice", "subscription", "renewal", "billing",
    "payment", "charged", "recurring", "monthly plan", "annual plan",
    "your plan", "membership", "trial ending", "trial expires",
    "auto-renew", "cancellation confirmed", "cancelled", "bill",
    "statement", "autopay", "plan summary", "account charge",
    "monthly charge", "service fee", "your payment", "payment received",
    "billing summary", "plan renewed", "upcoming charge",
]
