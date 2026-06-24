"""
증권사 API 추상화 레이어.
KIS (한국투자증권)와 토스증권 모두 OAuth 2.0 Client Credentials 방식을 사용하므로
구조가 거의 동일하다. provider= 설정으로 런타임에 교체 가능.
"""
from __future__ import annotations

import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import FinanceDataReader as fdr

logger = logging.getLogger(__name__)


# ─── 공통 데이터 클래스 ───────────────────────────────────────────────────────

@dataclass
class PriceInfo:
    symbol: str
    name: str
    price: int
    open: int
    high: int
    low: int
    volume: int
    change_pct: float
    per: float
    pbr: float
    market_cap: int      # 원 단위


@dataclass
class Position:
    symbol: str
    name: str
    quantity: int
    avg_price: float
    current_price: int
    pnl_pct: float


@dataclass
class Balance:
    cash: int
    total_value: int
    positions: List[Position]


@dataclass
class OrderResult:
    success: bool
    order_id: str
    message: str


# ─── 추상 베이스 ──────────────────────────────────────────────────────────────

class BaseProvider(ABC):
    @abstractmethod
    def get_price(self, symbol: str) -> PriceInfo: ...

    @abstractmethod
    def get_balance(self) -> Balance: ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: int, price: int = 0) -> OrderResult: ...

    def get_ohlcv(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """FinanceDataReader로 일봉 OHLCV 조회 (KIS·토스 공통)"""
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        df = fdr.DataReader(symbol, start, end)
        df.columns = [c.lower() for c in df.columns]
        cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[cols].dropna()


# ─── 한국투자증권 ─────────────────────────────────────────────────────────────

class KISProvider(BaseProvider):
    REAL_URL = "https://openapi.koreainvestment.com:9443"
    PAPER_URL = "https://openapivts.koreainvestment.com:29443"

    def __init__(self, app_key: str, app_secret: str, account_no: str, is_paper: bool = True):
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.is_paper = is_paper
        self.base_url = self.PAPER_URL if is_paper else self.REAL_URL
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._token_file = Path("kis_token.json")
        self._load_cached_token()

    def _load_cached_token(self):
        if self._token_file.exists():
            try:
                d = json.loads(self._token_file.read_text())
                self._token = d.get("token")
                exp = d.get("expires")
                if exp:
                    self._token_expires = datetime.fromisoformat(exp)
            except Exception:
                pass

    def _ensure_token(self):
        if self._token and self._token_expires and datetime.now() < self._token_expires:
            return
        resp = requests.post(
            f"{self.base_url}/oauth2/tokenP",
            json={"grant_type": "client_credentials",
                  "appkey": self.app_key, "appsecret": self.app_secret},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        expires_in = int(data.get("expires_in", 86400))
        self._token_expires = datetime.now() + pd.Timedelta(seconds=expires_in - 300)
        self._token_file.write_text(json.dumps({
            "token": self._token,
            "expires": self._token_expires.isoformat(),
        }))

    def _headers(self, tr_id: str) -> dict:
        self._ensure_token()
        return {
            "authorization": f"Bearer {self._token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "content-type": "application/json; charset=utf-8",
        }

    def get_price(self, symbol: str) -> PriceInfo:
        time.sleep(0.12)
        resp = requests.get(
            f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=self._headers("FHKST01010100"),
            params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": symbol},
            timeout=10,
        )
        resp.raise_for_status()
        o = resp.json().get("output", {})
        return PriceInfo(
            symbol=symbol,
            name=o.get("hts_kor_isnm", ""),
            price=int(o.get("stck_prpr", 0) or 0),
            open=int(o.get("stck_oprc", 0) or 0),
            high=int(o.get("stck_hgpr", 0) or 0),
            low=int(o.get("stck_lwpr", 0) or 0),
            volume=int(o.get("acml_vol", 0) or 0),
            change_pct=float(o.get("prdy_ctrt", 0) or 0),
            per=float(o.get("per", 0) or 0),
            pbr=float(o.get("pbr", 0) or 0),
            market_cap=int(o.get("hts_avls", 0) or 0) * 100_000_000,
        )

    def get_balance(self) -> Balance:
        tr_id = "VTTC8434R" if self.is_paper else "TTTC8434R"
        cano = self.account_no[:8]
        acnt_prdt = self.account_no[8:] if len(self.account_no) > 8 else "01"
        resp = requests.get(
            f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
            headers=self._headers(tr_id),
            params={
                "CANO": cano, "ACNT_PRDT_CD": acnt_prdt,
                "AFHR_FLPR_YN": "N", "OFL_YN": "", "INQR_DVSN": "02",
                "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
                "CTX_AREA_FK100": "", "CTX_AREA_NK100": "",
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        positions = [
            Position(
                symbol=item["pdno"], name=item["prdt_name"],
                quantity=int(item["hldg_qty"] or 0),
                avg_price=float(item["pchs_avg_pric"] or 0),
                current_price=int(item["prpr"] or 0),
                pnl_pct=float(item["evlu_pfls_rt"] or 0),
            )
            for item in data.get("output1", [])
            if int(item.get("hldg_qty", 0) or 0) > 0
        ]
        o2 = data.get("output2", [{}])[0]
        return Balance(
            cash=int(o2.get("dnca_tot_amt", 0) or 0),
            total_value=int(o2.get("tot_evlu_amt", 0) or 0),
            positions=positions,
        )

    def place_order(self, symbol: str, side: str, quantity: int, price: int = 0) -> OrderResult:
        if side.upper() == "BUY":
            tr_id = "VTTC0802U" if self.is_paper else "TTTC0802U"
        else:
            tr_id = "VTTC0801U" if self.is_paper else "TTTC0801U"
        cano = self.account_no[:8]
        acnt_prdt = self.account_no[8:] if len(self.account_no) > 8 else "01"
        ord_dvsn = "01" if price == 0 else "00"   # 01=시장가, 00=지정가
        try:
            resp = requests.post(
                f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash",
                headers=self._headers(tr_id),
                json={
                    "CANO": cano, "ACNT_PRDT_CD": acnt_prdt,
                    "PDNO": symbol, "ORD_DVSN": ord_dvsn,
                    "ORD_QTY": str(quantity), "ORD_UNPR": str(price),
                },
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            success = result.get("rt_cd", "1") == "0"
            return OrderResult(
                success=success,
                order_id=result.get("output", {}).get("ODNO", ""),
                message=result.get("msg1", ""),
            )
        except Exception as e:
            return OrderResult(success=False, order_id="", message=str(e))


# ─── 토스증권 ─────────────────────────────────────────────────────────────────

class TossProvider(BaseProvider):
    """
    토스증권 Open API (2026년 사전신청 → 정식오픈 예정)
    공식 문서: https://developers.tossinvest.com/docs
    실제 엔드포인트·파라미터는 문서를 참고하여 조정 필요.
    OAuth 2.0 Client Credentials 구조는 KIS와 동일.
    """
    BASE_URL = "https://openapi.tossinvest.com"

    def __init__(self, client_id: str, client_secret: str, account_no: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account_no = account_no
        self._token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._token_file = Path("toss_token.json")
        self._load_cached_token()

    def _load_cached_token(self):
        if self._token_file.exists():
            try:
                d = json.loads(self._token_file.read_text())
                self._token = d.get("token")
                exp = d.get("expires")
                if exp:
                    self._token_expires = datetime.fromisoformat(exp)
            except Exception:
                pass

    def _ensure_token(self):
        if self._token and self._token_expires and datetime.now() < self._token_expires:
            return
        resp = requests.post(
            f"{self.BASE_URL}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data["access_token"]
        expires_in = int(data.get("expires_in", 86400))
        self._token_expires = datetime.now() + pd.Timedelta(seconds=expires_in - 300)
        self._token_file.write_text(json.dumps({
            "token": self._token,
            "expires": self._token_expires.isoformat(),
        }))

    def _headers(self) -> dict:
        self._ensure_token()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def get_price(self, symbol: str) -> PriceInfo:
        time.sleep(0.12)
        resp = requests.get(
            f"{self.BASE_URL}/v1/quotations/price",
            headers=self._headers(),
            params={"symbol": symbol},
            timeout=10,
        )
        resp.raise_for_status()
        o = resp.json()
        return PriceInfo(
            symbol=symbol, name=o.get("name", ""),
            price=int(o.get("price", 0) or 0),
            open=int(o.get("openPrice", 0) or 0),
            high=int(o.get("highPrice", 0) or 0),
            low=int(o.get("lowPrice", 0) or 0),
            volume=int(o.get("volume", 0) or 0),
            change_pct=float(o.get("changeRate", 0) or 0),
            per=float(o.get("per", 0) or 0),
            pbr=float(o.get("pbr", 0) or 0),
            market_cap=int(o.get("marketCap", 0) or 0),
        )

    def get_balance(self) -> Balance:
        resp = requests.get(
            f"{self.BASE_URL}/v1/accounts/{self.account_no}/balance",
            headers=self._headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        positions = [
            Position(
                symbol=item["symbol"], name=item.get("name", ""),
                quantity=int(item["quantity"] or 0),
                avg_price=float(item["avgPrice"] or 0),
                current_price=int(item["currentPrice"] or 0),
                pnl_pct=float(item.get("pnlRate", 0) or 0),
            )
            for item in data.get("positions", [])
        ]
        return Balance(
            cash=int(data.get("cash", 0) or 0),
            total_value=int(data.get("totalValue", 0) or 0),
            positions=positions,
        )

    def place_order(self, symbol: str, side: str, quantity: int, price: int = 0) -> OrderResult:
        payload: dict = {
            "accountNo": self.account_no,
            "symbol": symbol,
            "side": side.upper(),
            "quantity": quantity,
            "orderType": "MARKET" if price == 0 else "LIMIT",
        }
        if price > 0:
            payload["price"] = price
        try:
            resp = requests.post(
                f"{self.BASE_URL}/v1/orders",
                headers=self._headers(),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return OrderResult(success=True, order_id=data.get("orderId", ""), message="주문 완료")
        except Exception as e:
            return OrderResult(success=False, order_id="", message=str(e))


# ─── 팩토리 ──────────────────────────────────────────────────────────────────

def _make_kis(cfg) -> KISProvider:
    return KISProvider(cfg.kis_app_key, cfg.kis_app_secret, cfg.kis_account_no,
                       getattr(cfg, "kis_is_paper", True))

def _make_toss(cfg) -> Optional[TossProvider]:
    if cfg.toss_client_id and cfg.toss_client_secret:
        return TossProvider(cfg.toss_client_id, cfg.toss_client_secret, cfg.toss_account_no)
    return None

def create_data_provider(cfg) -> BaseProvider:
    """시세 조회·차트·백테스팅용 프로바이더 (DATA_PROVIDER 설정 기준)"""
    if getattr(cfg, "data_provider", "kis") == "toss":
        toss = _make_toss(cfg)
        if toss:
            return toss
    return _make_kis(cfg)

def create_trade_provider(cfg) -> BaseProvider:
    """실제 주문·잔고 조회용 프로바이더 (TRADE_PROVIDER 설정 기준)
    토스 키가 미입력이면 KIS로 폴백.
    """
    if getattr(cfg, "trade_provider", "toss") == "toss":
        toss = _make_toss(cfg)
        if toss:
            return toss
        logger.warning("토스증권 키 미입력 → KIS로 폴백 (주문 기능)")
    return _make_kis(cfg)

# 하위 호환용
def create_provider(cfg) -> BaseProvider:
    return create_data_provider(cfg)
