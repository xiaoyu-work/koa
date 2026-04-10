"""
User Tools - User profile and connected account lookup

These tools use CredentialStore via AgentToolContext.credentials
to list connected accounts and user profile information.
"""

import logging

from koa.models import AgentToolContext

logger = logging.getLogger(__name__)


async def get_user_accounts_executor(args: dict, context: AgentToolContext = None) -> str:
    """Get user's connected accounts from CredentialStore."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    cred_store = context.credentials
    if not cred_store:
        return "Error: Credential store not configured"

    try:
        accounts = await cred_store.list(context.tenant_id)

        if not accounts:
            return "You don't have any connected accounts yet."

        output = []
        for account in accounts:
            service = account.get("service", "unknown")
            account_name = account.get("account_name", "primary")
            creds = account.get("credentials", {})
            email = creds.get("account_identifier") or creds.get("email", "")

            label = f"- {service}"
            if account_name != "primary":
                label += f" ({account_name})"
            if email:
                label += f": {email}"
            output.append(label)

        return f"Connected accounts ({len(accounts)}):\n" + "\n".join(output)

    except Exception as e:
        logger.error(f"Error getting user accounts: {e}", exc_info=True)
        return f"Error retrieving account information: {e}"


async def get_user_profile_executor(args: dict, context: AgentToolContext = None) -> str:
    """Get user's profile information from metadata."""
    if not context or not context.tenant_id:
        return "Error: User ID not available"

    # Profile can be provided via context metadata by the application
    profile = context.metadata.get("user_profile")

    if not profile:
        return "No profile information found."

    output = []
    if profile.get("first_name"):
        name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        output.append(f"Name: {name}")
    if profile.get("email"):
        output.append(f"Email: {profile.get('email')}")
    if profile.get("phone"):
        output.append(f"Phone: {profile.get('phone')}")
    if profile.get("timezone"):
        output.append(f"Timezone: {profile.get('timezone')}")

    if not output:
        return "Profile exists but no details available."

    return "User profile:\n" + "\n".join(output)


GET_USER_ACCOUNTS_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}

GET_USER_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
}


# ---------------------------------------------------------------------------
# update_user_profile — write tool
# ---------------------------------------------------------------------------


async def update_user_profile_executor(args: dict, context: AgentToolContext = None) -> str:
    """Save or update the user's profile information via koi-backend API."""
    import httpx

    if not context or not context.tenant_id:
        return "Error: User ID not available"

    category = args.get("category", "")
    data = args.get("data", {})

    if not category or not data:
        return "Error: 'category' and 'data' are required."

    VALID = {
        "identity",
        "addresses",
        "work",
        "finance",
        "travel",
        "lifestyle",
        "relationships",
        "services",
        "memories",
        "ai_preferences",
    }
    if category not in VALID:
        return f"Error: Invalid category '{category}'. Must be one of: {', '.join(sorted(VALID))}"

    meta = context.metadata or {}
    koiai_url = meta.get("koiai_url")
    service_key = meta.get("service_key", "")

    if not koiai_url:
        return "Error: Backend URL not configured — cannot save profile."

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{koiai_url}/api/profile/update-field",
                json={
                    "tenant_id": context.tenant_id,
                    "category": category,
                    "data": data,
                },
                headers={"X-Service-Key": service_key} if service_key else {},
            )
            if resp.status_code == 200:
                result = resp.json()
                return f"Profile {result.get('status', 'updated')}: {category} saved successfully."
            else:
                return f"Error saving profile: {resp.status_code} {resp.text}"
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}", exc_info=True)
        return f"Error saving profile: {e}"


UPDATE_USER_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": "Profile category to update. One of: identity, addresses, work, finance, travel, lifestyle, relationships, services, memories, ai_preferences",
            "enum": [
                "identity",
                "addresses",
                "work",
                "finance",
                "travel",
                "lifestyle",
                "relationships",
                "services",
                "memories",
                "ai_preferences",
            ],
        },
        "data": {
            "type": "object",
            "description": "Data to save. Structure depends on category. Examples:\n"
            '- addresses: [{"label": "home", "address": "123 Main St", "city": "Seattle", "state": "WA"}]\n'
            '- work: {"company": "Microsoft", "title": "Engineer", "office": "Building 16"}\n'
            '- identity: {"first_name": "John", "last_name": "Doe"}\n'
            '- ai_preferences: {"language": "zh", "verbosity": "brief"}',
        },
    },
    "required": ["category", "data"],
}
