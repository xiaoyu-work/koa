"""Subscription detection and tracking provider."""

from .detector import SubscriptionDetector
from .known_services import KNOWN_SERVICES

__all__ = ["SubscriptionDetector", "KNOWN_SERVICES"]
