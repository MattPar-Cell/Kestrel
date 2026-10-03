"""Signal notifications: trade recommendations and the weekly summary.

`messages` turns intents and summaries into text and is pure. `signal` sends
text through a signal-cli-rest-api container and is the only I/O here.
"""

from kestrel.notify.messages import (
    Recommendation,
    format_money,
    format_recommendations,
    format_weekly_summary,
)
from kestrel.notify.signal import SignalError, SignalNotifier

__all__ = [
    "Recommendation",
    "SignalError",
    "SignalNotifier",
    "format_money",
    "format_recommendations",
    "format_weekly_summary",
]
