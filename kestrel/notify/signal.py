"""Send messages through signal-cli-rest-api.

Signal has no official bot API. The usual route — and the one this assumes — is
the `bbernhard/signal-cli-rest-api` Docker container, linked to a Signal account
as a secondary device, exposing `POST /v2/send`. The container holds the Signal
keys; Kestrel only needs its URL and the sending number.

No retries: a failed notification is logged by the caller and surfaced, never
silently re-sent, because a duplicated trade message is worse than a late one.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from kestrel.config.settings import Settings


class SignalError(RuntimeError):
    pass


class SignalNotifier:
    def __init__(
        self,
        api_url: str,
        sender: str,
        recipients: Sequence[str],
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        if not recipients:
            raise ValueError("at least one Signal recipient is required")
        self.api_url = api_url.rstrip("/")
        self.sender = sender
        self.recipients = list(recipients)
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs: object) -> SignalNotifier:
        if not settings.signal_configured:
            raise ValueError("set KESTREL_SIGNAL_SENDER and KESTREL_SIGNAL_RECIPIENTS")
        assert settings.signal_sender is not None
        return cls(
            settings.signal_api_url,
            settings.signal_sender,
            settings.signal_recipients,
            **kwargs,  # type: ignore[arg-type]
        )

    async def send(self, text: str) -> None:
        payload = {"message": text, "number": self.sender, "recipients": self.recipients}
        try:
            resp = await self._client.post(f"{self.api_url}/v2/send", json=payload)
        except httpx.HTTPError as e:
            raise SignalError(f"could not reach signal-cli-rest-api at {self.api_url}: {e}") from e
        if resp.status_code >= 300:
            raise SignalError(f"Signal send failed ({resp.status_code}): {resp.text[:200]}")

    async def aclose(self) -> None:
        await self._client.aclose()
