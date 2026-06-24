from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from analysis.analyst import ClaudeAnalyst
from config import config
from data.provider import BaseProvider
from indicators.technical import TechnicalSignals, calculate_signals

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    symbol: str
    name: str
    action: str            # BUY / SELL / HOLD / SKIP
    confidence: float
    price: int
    quantity: int
    target_price: Optional[int]
    stop_loss: Optional[int]
    hold_period: str
    summary: str
    key_factors: List[str]
    signals: Optional[TechnicalSignals]
    claude_used: bool
    timestamp: datetime = field(default_factory=datetime.now)


class DecisionEngine:
    def __init__(self, provider: BaseProvider, analyst: ClaudeAnalyst):
        self.provider = provider
        self.analyst = analyst
        self._day_start_value: int = 0

    def evaluate(
        self,
        symbol: str,
        name: str,
        portfolio_value: int,
        current_positions: Optional[List[dict]] = None,
    ) -> Optional[TradeDecision]:
        try:
            price_info = self.provider.get_price(symbol)
            df = self.provider.get_ohlcv(symbol, "2023-01-01")
            if len(df) < 60:
                logger.debug("%s: 데이터 부족 (%d일)", symbol, len(df))
                return None

            signals = calculate_signals(df, symbol)
            fundamental = {
                "per": price_info.per,
                "pbr": price_info.pbr,
                "market_cap_b": price_info.market_cap // 100_000_000,
            }

            analysis = self.analyst.decide(
                symbol=symbol,
                name=name,
                price=price_info.price,
                signals=signals,
                fundamental=fundamental,
            )

            action = analysis.get("action", "HOLD")
            quantity = 0
            if action == "BUY" and price_info.price > 0:
                n_pos = len(current_positions or [])
                if n_pos < config.max_positions:
                    pos_value = portfolio_value * config.position_size_pct
                    quantity = max(1, int(pos_value / price_info.price))

            return TradeDecision(
                symbol=symbol,
                name=name,
                action=action,
                confidence=float(analysis.get("confidence", 0.5)),
                price=price_info.price,
                quantity=quantity,
                target_price=analysis.get("target_price"),
                stop_loss=analysis.get("stop_loss"),
                hold_period=analysis.get("hold_period", "없음"),
                summary=analysis.get("summary", ""),
                key_factors=analysis.get("key_factors", []),
                signals=signals,
                claude_used=bool(analysis.get("claude_used", False)),
            )
        except Exception as e:
            logger.error("evaluate[%s]: %s", symbol, e)
            return None

    def check_stop_losses(self, positions: List[dict]) -> List[dict]:
        """손절 조건 충족 포지션 반환"""
        triggers = []
        for pos in positions:
            avg = float(pos.get("avg_price", 0))
            cur = float(pos.get("current_price", 0))
            if avg <= 0:
                continue
            loss_pct = (cur - avg) / avg
            if loss_pct <= -config.stop_loss_pct:
                triggers.append({**pos, "reason": f"손절 발동 ({loss_pct*100:.1f}%)"})
        return triggers

    def is_daily_loss_exceeded(self, current_value: int) -> bool:
        if self._day_start_value == 0:
            self._day_start_value = current_value
            return False
        loss = (current_value - self._day_start_value) / self._day_start_value
        return loss <= -config.max_daily_loss_pct

    def reset_day(self, current_value: int):
        self._day_start_value = current_value
