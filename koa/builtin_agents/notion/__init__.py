"""
Notion integration for Koa

NotionAgent handles all Notion operations (search, read, create, update)
via an internal mini ReAct loop.
"""

from .agent import NotionAgent

__all__ = [
    "NotionAgent",
]
