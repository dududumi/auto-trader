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

# 미국 주식 티커 → 이름 (S&P 500 주요 종목 + 인기 ETF)
_US_NAMES: dict[str, str] = {
    # 빅테크
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "GOOG": "Alphabet C",
    "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA", "TSLA": "Tesla",
    "NFLX": "Netflix", "UBER": "Uber", "LYFT": "Lyft", "SNAP": "Snap",
    "TWTR": "Twitter", "PINS": "Pinterest", "RBLX": "Roblox", "U": "Unity",
    # 반도체
    "AMD": "AMD", "INTC": "Intel", "QCOM": "Qualcomm", "AVGO": "Broadcom",
    "AMAT": "Applied Materials", "LRCX": "Lam Research", "KLAC": "KLA Corp",
    "MRVL": "Marvell", "MPWR": "Monolithic Power", "ON": "ON Semiconductor",
    "TXN": "Texas Instruments", "MU": "Micron", "WDC": "Western Digital",
    "STX": "Seagate", "AEHR": "Aehr Test", "WOLF": "Wolfspeed",
    "TSM": "TSMC", "ASML": "ASML", "SMCI": "Super Micro",
    # 소프트웨어·클라우드
    "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe", "NOW": "ServiceNow",
    "WDAY": "Workday", "SNOW": "Snowflake", "DDOG": "Datadog", "MDB": "MongoDB",
    "NET": "Cloudflare", "ZS": "Zscaler", "CRWD": "CrowdStrike", "PANW": "Palo Alto",
    "FTNT": "Fortinet", "OKTA": "Okta", "PLTR": "Palantir", "PATH": "UiPath",
    "AI": "C3.ai", "BBAI": "BigBear.ai", "SOUN": "SoundHound",
    # 핀테크·결제
    "PYPL": "PayPal", "SQ": "Block", "V": "Visa", "MA": "Mastercard",
    "AXP": "American Express", "AFRM": "Affirm", "SOFI": "SoFi",
    "COIN": "Coinbase", "HOOD": "Robinhood",
    # 금융·은행
    "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo", "C": "Citigroup",
    "BLK": "BlackRock", "SCHW": "Charles Schwab", "BRK.B": "Berkshire Hathaway B",
    # 헬스케어·바이오
    "JNJ": "J&J", "PFE": "Pfizer", "MRK": "Merck", "ABBV": "AbbVie",
    "AMGN": "Amgen", "GILD": "Gilead", "BIIB": "Biogen", "REGN": "Regeneron",
    "MRNA": "Moderna", "BNTX": "BioNTech", "LLY": "Eli Lilly",
    "UNH": "UnitedHealth", "CVS": "CVS Health", "CI": "Cigna",
    # 소비재·유통
    "AMZN": "Amazon", "WMT": "Walmart", "TGT": "Target", "COST": "Costco",
    "HD": "Home Depot", "LOW": "Lowe's", "SBUX": "Starbucks", "MCD": "McDonald's",
    "NKE": "Nike", "LULU": "Lululemon", "TJX": "TJX", "ROST": "Ross Stores",
    # 에너지
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "SLB": "SLB", "EOG": "EOG Resources",
    # 통신
    "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile",
    # 미디어·엔터
    "DIS": "Disney", "CMCSA": "Comcast", "PARA": "Paramount", "WBD": "Warner Bros",
    "SPOT": "Spotify",
    # 항공·우주·방산
    "BA": "Boeing", "LMT": "Lockheed Martin", "RTX": "Raytheon",
    "NOC": "Northrop Grumman", "GD": "General Dynamics",
    "AAL": "American Airlines", "UAL": "United Airlines", "DAL": "Delta",
    "SPCE": "Virgin Galactic", "RKT": "Rocket Companies",
    # 전기차·친환경
    "RIVN": "Rivian", "LCID": "Lucid", "NIO": "NIO", "XPEV": "XPeng",
    "LI": "Li Auto", "F": "Ford", "GM": "GM",
    "ENPH": "Enphase", "SEDG": "SolarEdge", "FSLR": "First Solar",
    # 부동산·리츠
    "AMT": "American Tower", "PLD": "Prologis", "EQIX": "Equinix",
    # ETF
    "SPY": "S&P 500 ETF", "QQQ": "NASDAQ 100 ETF", "DIA": "Dow Jones ETF",
    "IWM": "Russell 2000 ETF", "VTI": "Vanguard Total Market",
    "VOO": "Vanguard S&P 500", "ARKK": "ARK Innovation", "SOXL": "반도체 3x ETF",
    "TQQQ": "NASDAQ 3x ETF", "SQQQ": "NASDAQ -3x ETF",
    "GLD": "Gold ETF", "SLV": "Silver ETF", "USO": "Oil ETF",
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


def search_us(query: str, limit: int = 8) -> list[dict]:
    """미국 주식 티커/이름 검색."""
    q = query.strip().upper()
    if not q:
        return []
    results = []
    for ticker, name in _US_NAMES.items():
        if q in ticker or q in name.upper():
            results.append({"code": ticker, "name": name, "label": f"{name} ({ticker})"})
        if len(results) >= limit:
            break
    return results


def is_us_symbol(symbol: str) -> bool:
    """미국 주식 티커 여부 (숫자 6자리가 아니면 US로 간주)."""
    return not symbol.strip().isdigit()
