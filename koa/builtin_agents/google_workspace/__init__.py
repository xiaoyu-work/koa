"""
Google Workspace integration for Koa

GoogleWorkspaceAgent handles all Google Workspace operations
(Drive search, Docs read/create, Sheets read/write) via an internal
mini ReAct loop.
"""

from .agent import GoogleWorkspaceAgent

__all__ = [
    "GoogleWorkspaceAgent",
]
