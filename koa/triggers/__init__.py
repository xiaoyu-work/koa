"""
Koa Triggers — Proactive trigger system.

Supports schedule (cron/interval/one-time), event (Redis Streams),
and condition (periodic polling) triggers with dual execution paths:
OrchestratorExecutor (LLM-driven) and custom executors (deterministic).
"""

from .callback import CallbackNotification
from .cron.delivery import CronDeliveryHandler
from .cron.executor import CronExecutor
from .cron.models import (
    CronEvent,
    CronJob,
    CronJobCreate,
    CronJobPatch,
    CronJobState,
    CronRunEntry,
)
from .cron.run_log import CronRunLog
from .cron.service import CronService
from .cron.store import CronJobStore
from .email_handler import EmailEventHandler
from .engine import TriggerEngine
from .event_bus import Event, EventBus
from .executor import OrchestratorExecutor
from .models import (
    ActionConfig,
    ActionResult,
    Task,
    TaskStatus,
    TriggerConfig,
    TriggerContext,
    TriggerType,
)
from .notification import PushNotification, SMSNotification
from .pipeline import PipelineExecutor

__all__ = [
    # Models
    "Task",
    "TaskStatus",
    "TriggerConfig",
    "TriggerType",
    "ActionConfig",
    "TriggerContext",
    "ActionResult",
    # Engine
    "TriggerEngine",
    # EventBus
    "EventBus",
    "Event",
    # Executors
    "OrchestratorExecutor",
    "PipelineExecutor",
    # Email
    "EmailEventHandler",
    # Notifications
    "SMSNotification",
    "PushNotification",
    "CallbackNotification",
    # Cron
    "CronJob",
    "CronJobState",
    "CronRunEntry",
    "CronEvent",
    "CronJobCreate",
    "CronJobPatch",
    "CronService",
    "CronJobStore",
    "CronRunLog",
    "CronExecutor",
    "CronDeliveryHandler",
]
