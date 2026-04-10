"""
Action History Tool - Query past tool executions for context.

Allows the LLM to recall recent actions the user has performed,
enabling contextual conversations like "what did I do yesterday?"
"""

import json
import logging

from koa.models import AgentToolContext

logger = logging.getLogger(__name__)


async def recall_recent_actions_executor(args: dict, context: AgentToolContext = None) -> str:
    """Query recent tool call history for the current user."""
    limit = args.get("limit", 10)
    agent_filter = args.get("agent_name")

    if not context or not context.tenant_id:
        return "Error: No tenant context available."

    db = context.metadata.get("database") if context.metadata else None
    if not db:
        return "No action history available."

    try:
        if agent_filter:
            rows = await db.fetch(
                """
                SELECT tool_name, agent_name, summary, success,
                       duration_ms, created_at
                FROM tool_call_history
                WHERE tenant_id = $1 AND agent_name = $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                context.tenant_id,
                agent_filter,
                limit,
            )
        else:
            rows = await db.fetch(
                """
                SELECT tool_name, agent_name, summary, success,
                       duration_ms, created_at
                FROM tool_call_history
                WHERE tenant_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                context.tenant_id,
                limit,
            )

        if not rows:
            return "No recent actions found."

        lines = []
        for r in rows:
            ts = r["created_at"].strftime("%Y-%m-%d %H:%M") if r["created_at"] else "?"
            status = "\u2713" if r["success"] else "\u2717"
            agent = r["agent_name"] or "builtin"
            summary = r["summary"] or r["tool_name"]
            lines.append(f"[{ts}] {status} {agent}: {summary}")

        return f"Recent actions ({len(rows)}):\n" + "\n".join(lines)

    except Exception as e:
        logger.error(f"Failed to query action history: {e}")
        return f"Error retrieving action history: {e}"


async def save_tool_call_history(db, tenant_id: str, tool_calls: list) -> None:
    """Persist tool call records to the database.

    Called by the orchestrator after ReAct loop completion.
    Filters out internal tools (complete_task, generate_plan).
    """
    INTERNAL_TOOLS = {"complete_task", "generate_plan", "recall_memory", "recall_recent_actions"}

    for tc in tool_calls:
        name = tc.get("name", "")
        if name in INTERNAL_TOOLS:
            continue

        try:
            await db.execute(
                """
                INSERT INTO tool_call_history
                    (tenant_id, tool_name, agent_name, summary, args_summary,
                     success, result_status, result_chars, duration_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                tenant_id,
                name,
                tc.get("agent_name"),
                _make_summary(name, tc.get("args_summary", {})),
                json.dumps(tc.get("args_summary", {})),
                tc.get("success", True),
                tc.get("result_status"),
                tc.get("result_chars", 0),
                tc.get("duration_ms", 0),
            )
        except Exception as e:
            logger.error(f"Failed to save tool call '{name}': {e}")


def _make_summary(tool_name: str, args: dict) -> str:
    """Generate a human-readable one-line summary for an action."""
    # Agent-tools (PascalCase names) use task_instruction if available
    if tool_name and tool_name[0].isupper():
        instruction = args.get("task_instruction", "")
        if instruction:
            return instruction[:200]
        return f"Used {tool_name}"

    # Builtin tools - extract the most relevant argument
    for key in ("query", "url", "subject", "title", "name", "event_id", "message"):
        if key in args:
            val = str(args[key])
            return f"{tool_name}: {val[:150]}"

    return tool_name


RECALL_RECENT_ACTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "description": "Maximum number of recent actions to return",
            "default": 10,
        },
        "agent_name": {
            "type": "string",
            "description": "Filter by agent name (e.g. 'EmailAgent', 'CalendarAgent'). Omit to see all.",
        },
    },
    "required": [],
}
