from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from data.provider import BaseProvider
from decision.engine import TradeDecision

logger = logging.getLogger(__name__)

HISTORY_FILE = Path("order_history.json")


@dataclass
class OrderRecord:
    order_id: str
    symbol: str
    name: str
    side: str           # BUY / SELL
    quantity: int
    price: int
    status: str         # success / failed
    reason: str
    claude_used: bool
    timestamp: str


class OrderManager:
    def __init__(self, provider: BaseProvider, paper_trading: bool = True):
        self.provider = provider
        self.paper_trading = paper_trading
        self._history: List[OrderRecord] = _load_history()

    def execute(self, decision: TradeDecision) -> Optional[OrderRecord]:
        if decision.action not in ("BUY", "SELL"):
            return None
        if decision.action == "BUY" and decision.quantity <= 0:
            return None

        mode = "PAPER" if self.paper_trading else "REAL"
        logger.info("[%s] %s %s x%d @ %s원",
                    mode, decision.action, decision.symbol,
                    decision.quantity, f"{decision.price:,}")

        result = self.provider.place_order(
            symbol=decision.symbol,
            side=decision.action,
            quantity=decision.quantity,
            price=0,   # 시장가
        )

        record = OrderRecord(
            order_id=result.order_id or f"LOCAL_{datetime.now():%Y%m%d%H%M%S}",
            symbol=decision.symbol,
            name=decision.name,
            side=decision.action,
            quantity=decision.quantity,
            price=decision.price,
            status="success" if result.success else "failed",
            reason=decision.summary[:120],
            claude_used=decision.claude_used,
            timestamp=datetime.now().isoformat(),
        )
        self._history.append(record)
        _save_history(self._history)

        if not result.success:
            logger.warning("주문 실패 [%s]: %s", decision.symbol, result.message)
        return record

    def get_history(self, limit: int = 100) -> List[OrderRecord]:
        return list(reversed(self._history[-limit:]))


def _load_history() -> List[OrderRecord]:
    if HISTORY_FILE.exists():
        try:
            return [OrderRecord(**r) for r in json.loads(HISTORY_FILE.read_text())]
        except Exception:
            pass
    return []


def _save_history(history: List[OrderRecord]):
    HISTORY_FILE.write_text(
        json.dumps([asdict(r) for r in history], ensure_ascii=False, indent=2)
    )
