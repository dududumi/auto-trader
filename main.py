"""
자동 매매 스케줄러
사용: python main.py

국내(KRX) / 미국(NYSE·NASDAQ) 정규장 시간에만 각 워치리스트를 스캔.
두 시장 모두 같은 프로세스에서 동작하며, 각자의 장 시간에만 활성화.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from config import config
from data.provider import create_data_provider, create_trade_provider
from analysis.analyst import ClaudeAnalyst
from decision.engine import DecisionEngine, TradeDecision
from execution.orders import OrderManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

KST = ZoneInfo("Asia/Seoul")
EST = ZoneInfo("America/New_York")   # DST 자동 반영 (EST/EDT)

SCAN_INTERVAL = 300   # 5분


# ─── 워치리스트 ───────────────────────────────────────────────────────────────

KRX_WATCHLIST: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035720": "카카오",
    "005380": "현대차",
    "035420": "NAVER",
    "051910": "LG화학",
    "006400": "삼성SDI",
    "207940": "삼성바이오로직스",
}

# 미국 주식 — 토스증권 API에서 지원하는 티커 형식 확인 후 조정
US_WATCHLIST: dict[str, str] = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "GOOGL": "Alphabet",
}


# ─── 시장 스케줄 ──────────────────────────────────────────────────────────────

@dataclass
class MarketWindow:
    name: str
    tz: ZoneInfo
    open_hhmm: str    # "09:00"
    close_hhmm: str   # "15:30"

    def is_open(self) -> bool:
        now = datetime.now(self.tz)
        if now.weekday() >= 5:   # 주말
            return False
        t = now.strftime("%H:%M")
        return self.open_hhmm <= t <= self.close_hhmm

    def now_str(self) -> str:
        return datetime.now(self.tz).strftime("%H:%M:%S %Z")


KRX_SCHEDULE = MarketWindow("국내(KRX)",  KST, "09:00", "15:25")
US_SCHEDULE  = MarketWindow("미국(NYSE)", EST, "09:30", "15:55")


def next_open_seconds() -> int:
    """두 시장 중 가장 빨리 열리는 시장까지 남은 초 (최대 60초 단위로 대기)"""
    return 60


# ─── 스캔 ─────────────────────────────────────────────────────────────────────

def run_scan(
    engine: DecisionEngine,
    order_mgr: OrderManager,
    watchlist: dict[str, str],
    market_name: str,
):
    try:
        balance = engine.provider.get_balance()
        portfolio_value = balance.total_value or 10_000_000
        current_positions = [
            {"symbol": p.symbol, "name": p.name,
             "quantity": p.quantity, "avg_price": p.avg_price,
             "current_price": p.current_price}
            for p in balance.positions
        ]
    except Exception as e:
        logger.error("[%s] 잔고 조회 실패: %s", market_name, e)
        portfolio_value = 10_000_000
        current_positions = []

    # 일일 손실 한도
    if engine.is_daily_loss_exceeded(portfolio_value):
        logger.warning("[%s] ⚠️ 일일 손실 한도 초과. 스캔 중단.", market_name)
        return

    # 손절 체크
    for pos in engine.check_stop_losses(current_positions):
        logger.warning("[%s] 손절: %s %s", market_name, pos["symbol"], pos["reason"])
        d = TradeDecision(
            symbol=pos["symbol"], name=pos["name"],
            action="SELL", confidence=1.0,
            price=pos["current_price"], quantity=pos["quantity"],
            target_price=None, stop_loss=None, hold_period="없음",
            summary=pos["reason"], key_factors=[],
            signals=None, claude_used=False,
        )
        order_mgr.execute(d)

    # 워치리스트 스캔
    for symbol, name in watchlist.items():
        decision = engine.evaluate(symbol, name, portfolio_value, current_positions)
        if decision is None:
            continue

        flag = "🤖" if decision.claude_used else "📐"
        logger.info("[%s] %s %s(%s) → %s %.0f%% | %s",
                    market_name, flag, name, symbol,
                    decision.action, decision.confidence * 100,
                    decision.summary[:60])

        if decision.action in ("BUY", "SELL"):
            rec = order_mgr.execute(decision)
            if rec:
                logger.info("[%s] %s 주문: %s",
                            market_name,
                            "✅" if rec.status == "success" else "❌",
                            rec.order_id)

        time.sleep(1)

    stats = engine.analyst.usage_stats
    logger.info("[%s] Claude %d회 / ₩%.0f",
                market_name, stats["calls"], stats["est_cost_krw"])


# ─── 메인 루프 ────────────────────────────────────────────────────────────────

def main():
    logger.info("=== AI 자동 매매 시작 ===")
    logger.info("증권사: %s", config.provider.upper())
    logger.info("국내 워치리스트: %d종목 | 미국 워치리스트: %d종목",
                len(KRX_WATCHLIST), len(US_WATCHLIST))
    _log_market_support()

    data_prov  = create_data_provider(config)
    trade_prov = create_trade_provider(config)
    analyst    = ClaudeAnalyst()
    engine     = DecisionEngine(data_prov, analyst)
    order_mgr  = OrderManager(trade_prov, paper_trading=False)

    krx_reset_done = False
    us_reset_done  = False

    while True:
        krx_open = KRX_SCHEDULE.is_open()
        us_open  = US_SCHEDULE.is_open()

        # 일일 손실 카운터 — 각 시장 개장 첫 번째 스캔에서 리셋
        if krx_open and not krx_reset_done:
            engine.reset_day(0)
            krx_reset_done = True
            logger.info("국내장 개장 — 일일 손실 카운터 초기화")
        if not krx_open:
            krx_reset_done = False

        if us_open and not us_reset_done:
            us_reset_done = True
            logger.info("미국장 개장")
        if not us_open:
            us_reset_done = False

        if krx_open:
            logger.info("--- 국내장 스캔 (%s) ---", KRX_SCHEDULE.now_str())
            run_scan(engine, order_mgr, KRX_WATCHLIST, "국내")

        if us_open:
            logger.info("--- 미국장 스캔 (%s) ---", US_SCHEDULE.now_str())
            run_scan(engine, order_mgr, US_WATCHLIST, "미국")

        if not krx_open and not us_open:
            krx_now = datetime.now(KST).strftime("%H:%M KST")
            us_now  = datetime.now(EST).strftime("%H:%M %Z")
            logger.info("장외 시간 — KST %s / NY %s | %d초 대기",
                        krx_now, us_now, SCAN_INTERVAL)

        time.sleep(SCAN_INTERVAL)


def _log_market_support():
    """현재 시각 기준으로 각 시장 상태 출력"""
    for sched in [KRX_SCHEDULE, US_SCHEDULE]:
        status = "🟢 개장중" if sched.is_open() else "🔴 장외"
        logger.info("%s: %s (%s ~ %s 현지시간) / 현재 %s",
                    sched.name, status,
                    sched.open_hhmm, sched.close_hhmm,
                    sched.now_str())


if __name__ == "__main__":
    main()
