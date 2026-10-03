"""Point-in-time portfolio snapshots and an append-only log of them.

A snapshot is what you *own*, not what you traded: cash plus holdings, each
marked at a price. The weekly summary is built from two of them (the first and
last in the week) plus the fills in between, so the log only has to be written
once a day — more often is harmless.

All amounts are in the account's base currency. Converting a USD-priced stock or
a BTC balance is the job of whatever builds the snapshot, not of this module, so
that every number downstream can be added to every other number.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Holding:
    symbol: str
    quantity: float
    #: Mark price, base currency.
    price: float
    #: Average cost per unit, base currency. None if the venue does not report it.
    avg_cost: float | None = None
    #: Where it is held, e.g. "trading212" or "kraken". Informational only.
    venue: str = ""

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError(f"{self.symbol}: price must be non-negative, got {self.price}")

    @property
    def value(self) -> float:
        return self.quantity * self.price

    @property
    def unrealised_pnl(self) -> float | None:
        if self.avg_cost is None:
            return None
        return (self.price - self.avg_cost) * self.quantity


@dataclass(frozen=True, slots=True)
class Snapshot:
    timestamp: datetime
    cash: float
    holdings: tuple[Holding, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("snapshot timestamp must be tz-aware")
        symbols = [h.symbol for h in self.holdings]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"duplicate symbols in snapshot: {sorted(symbols)}")

    @property
    def invested(self) -> float:
        return sum(h.value for h in self.holdings)

    @property
    def equity(self) -> float:
        return self.cash + self.invested

    def holding(self, symbol: str) -> Holding | None:
        return next((h for h in self.holdings if h.symbol == symbol), None)

    def to_json(self) -> str:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return json.dumps(d, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Snapshot:
        d = json.loads(line)
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            cash=float(d["cash"]),
            holdings=tuple(Holding(**h) for h in d["holdings"]),
        )


class SnapshotLog:
    """One JSON snapshot per line, append-only.

    JSON lines rather than parquet: a day's snapshot is a few hundred bytes, the
    file is human-readable when something looks wrong, and an append cannot
    corrupt earlier rows.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, snapshot: Snapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(snapshot.to_json() + "\n")

    def load(self, start: datetime | None = None, end: datetime | None = None) -> list[Snapshot]:
        """Snapshots with `start <= timestamp <= end`, oldest first."""
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            snaps = _parse(f)
        return sorted(
            (
                s
                for s in snaps
                if (start is None or s.timestamp >= start) and (end is None or s.timestamp <= end)
            ),
            key=lambda s: s.timestamp,
        )


def _parse(lines: Iterable[str]) -> list[Snapshot]:
    return [Snapshot.from_json(line) for line in lines if line.strip()]
