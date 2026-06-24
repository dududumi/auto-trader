"""
이벤트 드리븐 백테스팅 엔진.

- 룩어헤드 바이어스 없음: 각 날짜에서 해당 날짜까지의 데이터만 사용
- 실제 거래비용 반영: 수수료(0.015%) + 증권거래세(0.2%)
- KOSPI 벤치마크 비교
- 4가지 내장 전략 + 커스텀 전략 함수 지원
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

from indicators.technical import TechnicalSignals, calculate_signals

logger = logging.getLogger(__name__)

COMMISSION = 0.00015   # 매수·매도 수수료 0.015%
SELL_TAX = 0.0020      # 증권거래세 0.20% (매도 시)


# ─── 데이터 클래스 ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date: str
    symbol: str
    action: str        # BUY / SELL
    price: float
    quantity: int
    value: float       # 실제 지출(BUY) 또는 수령(SELL) 금액
    pnl: float = 0.0
    pnl_pct: float = 0.0
    reason: str = ""


@dataclass
class BacktestResult:
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annual_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: List[Trade] = field(default_factory=list)
    benchmark_return_pct: float = 0.0


# ─── 내장 전략 ────────────────────────────────────────────────────────────────

def _strategy_composite(df: pd.DataFrame, sigs: TechnicalSignals) -> str:
    """기본 복합전략: MACD + RSI + 이동평균"""
    score = sigs.composite_score
    if score >= 0.45 and sigs.rsi < 70:
        return "BUY"
    if score <= -0.35 or sigs.rsi > 78:
        return "SELL"
    return "HOLD"


def _strategy_macd_cross(df: pd.DataFrame, sigs: TechnicalSignals) -> str:
    """MACD 크로스오버 전략"""
    if sigs.macd_crossover == "golden" and sigs.rsi < 65:
        return "BUY"
    if sigs.macd_crossover == "death" or sigs.rsi > 75:
        return "SELL"
    return "HOLD"


def _strategy_golden_cross(df: pd.DataFrame, sigs: TechnicalSignals) -> str:
    """이동평균 골든크로스 추세추종"""
    if sigs.golden_cross is True and sigs.macd_hist > 0 and sigs.rsi < 70:
        return "BUY"
    if sigs.golden_cross is False:
        return "SELL"
    return "HOLD"


def _strategy_bollinger(df: pd.DataFrame, sigs: TechnicalSignals) -> str:
    """볼린저밴드 반전 전략"""
    if sigs.bb_position < 0.15 and sigs.rsi < 40:
        return "BUY"
    if sigs.bb_position > 0.85 and sigs.rsi > 60:
        return "SELL"
    return "HOLD"


BUILT_IN_STRATEGIES: dict[str, Callable] = {
    "기본 복합전략 (추천)": _strategy_composite,
    "MACD 크로스오버": _strategy_macd_cross,
    "골든크로스 추세추종": _strategy_golden_cross,
    "볼린저밴드 반전": _strategy_bollinger,
}


# ─── 백테스팅 엔진 ────────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(self, initial_capital: float = 10_000_000):
        self.initial_capital = initial_capital

    def run(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        strategy_name: str = "기본 복합전략 (추천)",
        strategy_fn: Optional[Callable] = None,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.07,
        max_positions: int = 10,
    ) -> BacktestResult:
        fn = strategy_fn or BUILT_IN_STRATEGIES.get(strategy_name, _strategy_composite)

        cash = self.initial_capital
        # symbol -> {qty, avg_price, buy_date}
        positions: dict[str, dict] = {}
        equity_records: list[dict] = []
        trades: List[Trade] = []

        # 룩백용 데이터 로드 (250일 여유)
        lookback = (pd.Timestamp(start_date) - pd.Timedelta(days=250)).strftime("%Y-%m-%d")
        data: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = fdr.DataReader(sym, lookback, end_date)
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                if len(df) >= 120:
                    data[sym] = df
            except Exception as e:
                logger.warning("데이터 로드 실패 [%s]: %s", sym, e)

        trading_days = pd.bdate_range(start_date, end_date)

        for current_date in trading_days:
            portfolio_value = cash

            for sym, full_df in data.items():
                df_slice = full_df[full_df.index <= current_date]
                if len(df_slice) < 60:
                    continue

                cur_price = float(df_slice["close"].iloc[-1])

                # 보유 포지션 평가
                if sym in positions:
                    portfolio_value += positions[sym]["qty"] * cur_price

                    # 손절 체크
                    avg = positions[sym]["avg_price"]
                    loss = (cur_price - avg) / avg
                    if loss <= -stop_loss_pct:
                        qty = positions[sym]["qty"]
                        net = cur_price * qty * (1 - SELL_TAX - COMMISSION)
                        cash += net
                        pnl = net - avg * qty * (1 + COMMISSION)
                        trades.append(Trade(
                            date=current_date.strftime("%Y-%m-%d"),
                            symbol=sym, action="SELL",
                            price=cur_price, quantity=qty, value=net,
                            pnl=pnl, pnl_pct=loss * 100, reason="손절",
                        ))
                        del positions[sym]
                        continue

                # 신호 계산
                try:
                    sigs = calculate_signals(df_slice, sym)
                    action = fn(df_slice, sigs)
                except Exception:
                    continue

                if action == "BUY" and sym not in positions and len(positions) < max_positions:
                    qty = int(portfolio_value * position_size_pct / cur_price)
                    if qty > 0:
                        cost = cur_price * qty * (1 + COMMISSION)
                        if cost <= cash:
                            cash -= cost
                            positions[sym] = {
                                "qty": qty,
                                "avg_price": cur_price,
                                "buy_date": current_date.strftime("%Y-%m-%d"),
                            }
                            trades.append(Trade(
                                date=current_date.strftime("%Y-%m-%d"),
                                symbol=sym, action="BUY",
                                price=cur_price, quantity=qty, value=cost,
                                reason=f"score={sigs.composite_score:+.2f}",
                            ))

                elif action == "SELL" and sym in positions:
                    pos = positions[sym]
                    proceeds = cur_price * pos["qty"]
                    net = proceeds * (1 - SELL_TAX - COMMISSION)
                    cash += net
                    pnl = net - pos["avg_price"] * pos["qty"] * (1 + COMMISSION)
                    pnl_pct = (cur_price / pos["avg_price"] - 1) * 100
                    trades.append(Trade(
                        date=current_date.strftime("%Y-%m-%d"),
                        symbol=sym, action="SELL",
                        price=cur_price, quantity=pos["qty"], value=net,
                        pnl=pnl, pnl_pct=pnl_pct, reason=f"score={sigs.composite_score:+.2f}",
                    ))
                    del positions[sym]

            equity_records.append({"date": current_date, "value": portfolio_value})

        # 기간 종료 시 잔여 포지션 청산
        for sym, pos in list(positions.items()):
            if sym in data:
                last_price = float(data[sym]["close"].iloc[-1])
                net = last_price * pos["qty"] * (1 - SELL_TAX - COMMISSION)
                cash += net
                pnl = net - pos["avg_price"] * pos["qty"] * (1 + COMMISSION)
                trades.append(Trade(
                    date=end_date, symbol=sym, action="SELL",
                    price=last_price, quantity=pos["qty"], value=net,
                    pnl=pnl, pnl_pct=(last_price / pos["avg_price"] - 1) * 100,
                    reason="기간 종료",
                ))

        equity = pd.Series(
            [r["value"] for r in equity_records],
            index=[r["date"] for r in equity_records],
            name="portfolio",
        )

        # KOSPI 벤치마크
        bench_ret = 0.0
        try:
            kospi = fdr.DataReader("KS11", start_date, end_date)
            bench_ret = float((kospi["Close"].iloc[-1] / kospi["Close"].iloc[0] - 1) * 100)
        except Exception:
            pass

        return _calc_metrics(
            strategy_name, symbols, start_date, end_date,
            self.initial_capital, equity, trades, bench_ret,
        )


def _calc_metrics(
    name: str,
    symbols: List[str],
    start: str,
    end: str,
    initial: float,
    equity: pd.Series,
    trades: List[Trade],
    bench_ret: float,
) -> BacktestResult:
    if equity.empty or len(equity) < 2:
        equity = pd.Series([initial, initial])

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_years = max(len(equity) / 252, 0.01)
    annual_ret = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1) * 100

    daily_rets = equity.pct_change().dropna()
    std = daily_rets.std()
    sharpe = float(daily_rets.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    down_std = daily_rets[daily_rets < 0].std()
    sortino = float(daily_rets.mean() / down_std * np.sqrt(252)) if down_std > 0 else 0.0

    rolling_max = equity.cummax()
    max_dd = float(abs(((equity - rolling_max) / rolling_max).min()) * 100)

    sell_trades = [t for t in trades if t.action == "SELL" and t.reason != "기간 종료"]
    profits = [t.pnl for t in sell_trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    win_rate = len(wins) / len(profits) * 100 if profits else 0.0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")

    return BacktestResult(
        strategy_name=name,
        symbols=symbols,
        start_date=start,
        end_date=end,
        initial_capital=initial,
        final_capital=float(equity.iloc[-1]),
        total_return_pct=round(total_ret, 2),
        annual_return_pct=round(annual_ret, 2),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown_pct=round(max_dd, 2),
        win_rate_pct=round(win_rate, 1),
        profit_factor=round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
        total_trades=len([t for t in trades if t.action == "BUY"]),
        equity_curve=equity,
        daily_returns=daily_rets,
        trades=trades,
        benchmark_return_pct=round(bench_ret, 2),
    )
