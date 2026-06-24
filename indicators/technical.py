"""
기술적 지표 계산 모듈.
외부 라이브러리 의존 없이 pandas/numpy만으로 구현.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TechnicalSignals:
    symbol: str
    current_price: float

    # RSI (14)
    rsi: float
    rsi_signal: str          # overbought / neutral / oversold

    # MACD (12, 26, 9)
    macd: float
    macd_signal_line: float
    macd_hist: float
    macd_crossover: Optional[str]   # golden / death / None

    # 이동평균
    sma_20: float
    sma_60: float
    sma_120: float
    ema_12: float
    ema_26: float
    ma_trend: str            # uptrend / downtrend / sideways
    golden_cross: Optional[bool]     # MA20 > MA60
    price_vs_ma20_pct: float         # 현재가 MA20 대비 %

    # 볼린저밴드 (20, 2σ)
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float       # 0.0(하단) ~ 1.0(상단)
    bb_squeeze: bool         # 밴드폭 수축 → 변동성 폭발 임박

    # 스토캐스틱 (14, 3, 3)
    stoch_k: float
    stoch_d: float

    # 거래량
    volume_ratio: float      # 당일 / 20일 평균

    # 종합점수 (-1.0 강매도 ~ +1.0 강매수)
    composite_score: float


def calculate_signals(df: pd.DataFrame, symbol: str) -> TechnicalSignals:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(close)

    # ── RSI (14) ─────────────────────────────────────────────────────────────
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])
    rsi_signal = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    sig_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - sig_line

    macd_crossover: Optional[str] = None
    if n >= 2:
        if hist.iloc[-2] < 0 and hist.iloc[-1] > 0:
            macd_crossover = "golden"
        elif hist.iloc[-2] > 0 and hist.iloc[-1] < 0:
            macd_crossover = "death"

    # ── 이동평균 ──────────────────────────────────────────────────────────────
    _mean = float(close.mean())
    sma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else _mean
    sma60 = float(close.rolling(60).mean().iloc[-1]) if n >= 60 else _mean
    sma120 = float(close.rolling(120).mean().iloc[-1]) if n >= 120 else _mean
    cur = float(close.iloc[-1])
    golden_cross: Optional[bool] = bool(sma20 > sma60) if n >= 60 else None
    price_vs_ma20_pct = round((cur / sma20 - 1) * 100, 2) if sma20 else 0.0

    ma_trend = "sideways"
    if n >= 25:
        sma20_5ago = float(close.rolling(20).mean().iloc[-6])
        if sma20 > sma20_5ago * 1.002:
            ma_trend = "uptrend"
        elif sma20 < sma20_5ago * 0.998:
            ma_trend = "downtrend"

    # ── 볼린저밴드 (20, 2σ) ──────────────────────────────────────────────────
    bb_mid_s = close.rolling(20).mean()
    bb_std_s = close.rolling(20).std()
    bb_upper = float((bb_mid_s + 2 * bb_std_s).iloc[-1])
    bb_mid_val = float(bb_mid_s.iloc[-1])
    bb_lower = float((bb_mid_s - 2 * bb_std_s).iloc[-1])
    bb_range = bb_upper - bb_lower
    bb_position = float((cur - bb_lower) / bb_range) if bb_range > 0 else 0.5

    # 스퀴즈: 현재 밴드폭이 최근 20일 평균 밴드폭보다 25% 이상 좁으면
    width_s = (bb_mid_s + 2 * bb_std_s) - (bb_mid_s - 2 * bb_std_s)
    avg_width = float(width_s.rolling(20).mean().iloc[-1]) if n >= 40 else float(width_s.mean())
    bb_squeeze = bb_range < avg_width * 0.75 if avg_width > 0 else False

    # ── 스토캐스틱 (14, 3, 3) ────────────────────────────────────────────────
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    hl_range = (high14 - low14).replace(0, np.nan)
    raw_k = 100 * (close - low14) / hl_range
    stoch_k = float(raw_k.rolling(3).mean().iloc[-1])
    stoch_d = float(raw_k.rolling(3).mean().rolling(3).mean().iloc[-1])

    # ── 거래량 비율 ───────────────────────────────────────────────────────────
    vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
    volume_ratio = float(volume.iloc[-1] / vol_ma20) if vol_ma20 > 0 else 1.0

    # ── 종합점수 계산 ─────────────────────────────────────────────────────────
    score = 0.0

    # RSI (±0.25)
    if rsi < 30:
        score += 0.25
    elif rsi > 70:
        score -= 0.25
    elif 40 <= rsi <= 60:
        score += 0.05

    # MACD (±0.30)
    if macd_crossover == "golden":
        score += 0.30
    elif macd_crossover == "death":
        score -= 0.30
    elif float(hist.iloc[-1]) > 0:
        score += 0.12
    else:
        score -= 0.12

    # 이동평균 골든/데드크로스 (±0.20)
    if golden_cross is True:
        score += 0.20
    elif golden_cross is False:
        score -= 0.20

    # 볼린저밴드 위치 (±0.10)
    if bb_position < 0.15:
        score += 0.10
    elif bb_position > 0.85:
        score -= 0.10

    # 거래량 확인 (±0.10)
    if volume_ratio > 1.5:
        if float(hist.iloc[-1]) > 0:
            score += 0.10
        else:
            score -= 0.05

    # MA 추세 (±0.05)
    if ma_trend == "uptrend":
        score += 0.05
    elif ma_trend == "downtrend":
        score -= 0.05

    return TechnicalSignals(
        symbol=symbol,
        current_price=cur,
        rsi=round(rsi, 2),
        rsi_signal=rsi_signal,
        macd=round(float(macd_line.iloc[-1]), 4),
        macd_signal_line=round(float(sig_line.iloc[-1]), 4),
        macd_hist=round(float(hist.iloc[-1]), 4),
        macd_crossover=macd_crossover,
        sma_20=round(sma20, 0),
        sma_60=round(sma60, 0),
        sma_120=round(sma120, 0),
        ema_12=round(float(ema12.iloc[-1]), 0),
        ema_26=round(float(ema26.iloc[-1]), 0),
        ma_trend=ma_trend,
        golden_cross=golden_cross,
        price_vs_ma20_pct=price_vs_ma20_pct,
        bb_upper=round(bb_upper, 0),
        bb_middle=round(bb_mid_val, 0),
        bb_lower=round(bb_lower, 0),
        bb_position=round(bb_position, 3),
        bb_squeeze=bb_squeeze,
        stoch_k=round(stoch_k, 2),
        stoch_d=round(stoch_d, 2),
        volume_ratio=round(volume_ratio, 2),
        composite_score=round(float(np.clip(score, -1.0, 1.0)), 3),
    )
