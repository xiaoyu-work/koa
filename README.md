# Koa

**Your proactive AI chief of staff** — triages email, manages calendar, tracks expenses & packages, controls smart home, plans trips, monitors subscriptions. Reaches you via AI glasses before you even think to ask.

### What Koa Does

| | Feature | Description |
|---|---|---|
| ✉️ | **Email** | Triages Gmail & Outlook, flags what's important, sends on your behalf |
| 📅 | **Calendar** | Manages Google Calendar & Outlook events, resolves conflicts, sets reminders |
| ☑️ | **Tasks** | Google Tasks, Microsoft To Do, Todoist — unified under one AI |
| 💳 | **Expenses** | Tracks spending, scans receipts, manages budgets |
| 📦 | **Packages** | Auto-detects tracking numbers from emails, monitors delivery status |
| 🏠 | **Smart Home** | Controls Philips Hue lights & Sonos speakers by voice |
| ✈️ | **Trips** | Builds itineraries, searches flights & hotels |
| 📍 | **Maps** | Directions, nearby places, commute estimates |
| 📓 | **Notion** | Reads and writes to your Notion workspace |
| ☁️ | **Storage** | Google Drive, OneDrive, Dropbox file management |
| 🔔 | **Subscriptions** | Detects recurring charges, alerts on renewals |
| 📰 | **Briefings** | Morning digest of weather, calendar, emails, reminders |
| 🖼️ | **Image Gen** | Creates images via DALL·E, Gemini, or Azure |
| ⚡ | **Proactive** | Learns your habits, reaches out before you ask |
| 🕶️ | **AI Glasses** | Delivers notifications and interacts through smart glasses |

### How It's Different

Koa is **proactive** — it doesn't wait for you to ask. It watches your email, detects important messages, tracks your packages, and pushes notifications to your phone or AI glasses. It has **true memory** across conversations and learns your preferences over time. 30+ specialized agents work together through a ReAct orchestrator with automatic model routing.

## Quick Start

### 1. Install

```bash
git clone https://github.com/xiaoyu-work/koa.git
cd koa
uv sync --extra openai        # or: --extra anthropic, --all-extras
```

### 2. Start

```bash
uv run koa serve
```

The API server starts at **http://localhost:8000**.

### 3. Chat

In another terminal:

```bash
uv run koa chat
```

### 4. Configure

Go to **http://localhost:8000/settings** or edit `config.yaml` directly:

1. **LLM Provider** - Choose your AI provider (OpenAI, Azure, Anthropic, etc.), enter API key, model name, and database URL
2. **OAuth Apps** *(optional)* - Add Google/Microsoft OAuth app credentials to enable one-click account connection
3. **Connect Accounts** *(optional)* - Connect Gmail, Outlook, Google Calendar, or Outlook Calendar

That's it. Go back to the chat and start talking.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message, get response |
| `/stream` | POST | Send message, stream response (SSE) |
| `/health` | GET | Liveness probe (version, auth status) |
| `/health/ready` | GET | Readiness probe (checks DB, LLM) |

```bash
# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Do I have any unread emails?"}'

# Stream (SSE)
curl -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Do I have any unread emails?"}'

# Multi-tenant
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "user_123", "message": "What is on my calendar today?"}'

# With API key authentication
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"message": "Do I have any unread emails?"}'
```

## Config

`config.yaml` is created automatically via the settings page. You can also create it manually:

```yaml
database: postgresql://user:pass@host:5432/dbname

llm:
  provider: openai          # openai / anthropic / azure / dashscope / gemini / ollama
  model: gpt-4o
  api_key: sk-...           # or omit to use provider's default env var

embedding:
  provider: openai
  model: text-embedding-3-small
```

| Field | Required | Description |
|-------|----------|-------------|
| `database` | Yes | PostgreSQL connection URL |
| `llm.provider` | Yes | LLM provider |
| `llm.model` | Yes | Model name |
| `llm.api_key` | No | API key (defaults to provider env var) |
| `llm.base_url` | No | Custom endpoint (required for Azure) |
| `embedding.provider` | Yes | Embedding provider |
| `embedding.model` | Yes | Embedding model name |
| `system_prompt` | No | System prompt / personality |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection URL (used in config via `${DATABASE_URL}`) |
| `KOA_API_KEY` | No | API key for endpoint authentication. **If not set, all endpoints are unauthenticated (dev mode only).** |
| `KOA_SERVICE_KEY` | No | Service key for internal endpoints |
| `KOA_CREDENTIAL_KEY` | No | AES-256 encryption key for credentials at rest. When set, OAuth tokens are encrypted in the database. |
| `KOA_CORS_ORIGINS` | No | Comma-separated allowed CORS origins (default: `http://localhost:3000,http://localhost:5173`) |

## Model Routing

Koa can automatically route each request to the best LLM based on task complexity. Simple tasks (greetings, quick lookups) go to cheap/fast models; complex tasks (trip planning, multi-agent workflows) go to stronger models. This can cut API costs significantly without sacrificing quality.

### How It Works

1. A lightweight **classifier** model scores every incoming request on a 1-100 complexity scale.
2. A set of **rules** maps score ranges to registered LLM providers.
3. The selected provider is used for the **entire** ReAct loop of that request (all turns).

### Setup

Add `llm_providers` and `model_routing` sections to your `config.yaml`:

```yaml
llm_providers:
  strong:
    provider: anthropic
    model: claude-sonnet-4-20250514
    api_key: ${ANTHROPIC_API_KEY}
  fast:
    provider: openai
    model: gpt-4o-mini
    api_key: ${OPENAI_API_KEY}
  cheap:
    provider: dashscope
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}

model_routing:
  enabled: true
  classifier_provider: fast     # which provider scores the request
  default_provider: fast        # fallback if classifier fails
  rules:
    - score_range: [1, 30]      # trivial: greetings, quick Q&A
      provider: cheap
    - score_range: [31, 70]     # standard: single-agent tasks
      provider: fast
    - score_range: [71, 100]    # complex: multi-agent, trip planning
      provider: strong
```

When `model_routing.enabled` is not set or `false`, Koa uses the single LLM defined in the `llm` section (the default behavior).

## License

[BSL 1.1](LICENSE)
