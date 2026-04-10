"""
Koa Providers - Email, Calendar, SMS, and Shipment providers

Providers receive credentials dicts directly. They do not query any database.
The calling code (agent or orchestrator) is responsible for fetching credentials
from CredentialStore and passing them to the provider.
"""

from .calendar.factory import CalendarProviderFactory
from .email.factory import EmailProviderFactory
from .email.resolver import AccountResolver
from .sms.factory import SMSProviderFactory

__all__ = [
    "EmailProviderFactory",
    "CalendarProviderFactory",
    "SMSProviderFactory",
    "AccountResolver",
]
