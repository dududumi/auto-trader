"""
종목코드 → 종목명 조회 유틸리티.
KRX 전체 종목은 FinanceDataReader로 로드(캐싱).
미국 주식은 자주 쓰는 티커를 내장하고, 없으면 티커 그대로 반환.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# 자주 쓰는 미국 주식 티커 → 이름
_US_NAMES: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA",
    "TSLA": "Tesla", "NFLX": "Netflix", "AMD": "AMD",
    "INTC": "Intel", "QCOM": "Qualcomm", "AVGO": "Broadcom",
    "TSM": "TSMC", "ASML": "ASML", "ORCL": "Oracle",
    "CRM": "Salesforce", "ADBE": "Adobe", "PYPL": "PayPal",
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "AMGN": "Amgen", "JNJ": "J&J", "PFE": "Pfizer",
    "SPY": "S&P500 ETF", "QQQ": "NASDAQ100 ETF", "DIA": "Dow ETF",
}


@lru_cache(maxsize=1)
def _krx_map() -> dict[str, str]:
    """KRX 전종목 코드→이름 딕셔너리 (프로세스당 1회 로드)"""
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX")[["Code", "Name"]].dropna()
        return dict(zip(df["Code"].astype(str).str.zfill(6), df["Name"]))
    except Exception as e:
        logger.warning("KRX 종목 목록 로드 실패: %s", e)
        return {}


def get_name(symbol: str) -> str:
    """종목코드 또는 티커로 종목명 반환. 없으면 symbol 그대로."""
    symbol = symbol.strip().upper()

    # 미국 주식 (알파벳으로만 구성)
    if symbol.isalpha():
        return _US_NAMES.get(symbol, symbol)

    # KRX (숫자 6자리)
    code = symbol.zfill(6)
    return _krx_map().get(code, symbol)


def get_name_with_code(symbol: str) -> str:
    """'삼성전자 (005930)' 형식으로 반환"""
    name = get_name(symbol)
    if name == symbol:
        return symbol
    return f"{name} ({symbol})"


def search_krx(query: str, limit: int = 10) -> list[dict]:
    """종목명 또는 코드로 검색. 자동완성용."""
    query = query.strip()
    if not query:
        return []
    results = []
    for code, name in _krx_map().items():
        if query in name or query in code:
            results.append({"code": code, "name": name, "label": f"{name} ({code})"})
        if len(results) >= limit:
            break
    return results
