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
    price: float         # KRX=원(정수), US=달러(소수 가능)
    open: float
    high: float
    low: float
    volume: int
    change_pct: float
    per: float
    pbr: float
    market_cap: int      # 원 단위


@dataclass
class Position:
    symbol: str
    name: str
    quantity: float        # 미국 소수점 매매(fractional shares) 대응
    avg_price: float
    current_price: float   # USD 종목 소수점 보존을 위해 float
    pnl_pct: float
    currency: str = "KRW"  # "KRW" 또는 "USD"


@dataclass
class Balance:
    cash: int
    total_value: int
    positions: List[Position]
    usd_market_value: float = 0.0  # USD 보유 평가금액 (달러)
    usd_to_krw: float = 0.0        # 환율 (1 USD = ? KRW)


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
        positions = []
        for item in data.get("output1", []):
            qty = float(item.get("hldg_qty", 0) or 0)
            if qty <= 0:
                continue
            avg_p = float(item.get("pchs_avg_pric", 0) or 0)
            cur_p = float(item.get("prpr", 0) or 0)
            pnl = (cur_p - avg_p) / avg_p * 100 if avg_p > 0 else 0.0
            positions.append(Position(
                symbol=item["pdno"], name=item["prdt_name"],
                quantity=qty,
                avg_price=avg_p,
                current_price=cur_p,
                pnl_pct=round(pnl, 4),
            ))
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
        self._account_seq: Optional[int] = None   # X-Tossinvest-Account 에 사용
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
            headers={"Content-Type": "application/x-www-form-urlencoded"},
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

    def _ensure_account_seq(self):
        if self._account_seq is not None:
            return
        resp = requests.get(
            f"{self.BASE_URL}/api/v1/accounts",
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=10,
        )
        resp.raise_for_status()
        for acc in resp.json().get("result", []):
            if acc.get("accountNo") == self.account_no:
                self._account_seq = acc["accountSeq"]
                return
        # 계좌번호 매칭 실패 시 첫 번째 계좌 사용
        accounts = resp.json().get("result", [])
        if accounts:
            self._account_seq = accounts[0]["accountSeq"]

    def _headers(self, with_account: bool = False) -> dict:
        self._ensure_token()
        h = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        if with_account:
            self._ensure_account_seq()
            if self._account_seq is not None:
                h["X-Tossinvest-Account"] = str(self._account_seq)
        return h

    def get_price(self, symbol: str) -> PriceInfo:
        from data.names import get_name
        time.sleep(0.12)
        resp = requests.get(
            f"{self.BASE_URL}/api/v1/prices",
            headers=self._headers(),
            params={"symbols": symbol},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json().get("result", [])
        o = result[0] if result else {}
        price = float(o.get("lastPrice", 0) or 0)
        name = (o.get("symbolName") or o.get("name") or "").strip() or get_name(symbol)
        return PriceInfo(
            symbol=symbol, name=name,
            price=price, open=0.0, high=0.0, low=0.0, volume=0,
            change_pct=0.0, per=0.0, pbr=0.0, market_cap=0,
        )

    def get_balance(self) -> Balance:
        headers = self._headers(with_account=True)
        # 보유 종목
        resp = requests.get(f"{self.BASE_URL}/api/v1/holdings", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("result", {})
        raw_positions = []
        for item in data.get("items", []):
            qty = float(item.get("quantity", 0) or 0)
            if qty <= 0:
                continue
            avg_p = float(item.get("averagePurchasePrice", 0) or 0)
            cur_p = float(item.get("lastPrice", 0) or 0)
            # API의 profitLoss.rate 단위가 불분명하므로 직접 계산 (avg_p 기준 수익률)
            pnl = (cur_p - avg_p) / avg_p * 100 if avg_p > 0 else 0.0
            raw_positions.append(Position(
                symbol=item["symbol"],
                name=item.get("name", ""),
                quantity=qty,
                avg_price=avg_p,
                current_price=cur_p,
                pnl_pct=round(pnl, 4),
                currency=item.get("currency", "KRW"),
            ))
        # holdings lastPrice보다 /api/v1/prices가 더 실시간 → 한 번 더 조회해 current_price 갱신
        if raw_positions:
            syms = [p.symbol for p in raw_positions]
            try:
                pr = requests.get(
                    f"{self.BASE_URL}/api/v1/prices",
                    headers=self._headers(),
                    params={"symbols": ",".join(syms)},
                    timeout=10,
                )
                if pr.status_code == 200:
                    price_map = {
                        r.get("symbol", ""): float(r.get("lastPrice", 0) or 0)
                        for r in pr.json().get("result", [])
                    }
                    for pos in raw_positions:
                        fresh = price_map.get(pos.symbol, 0)
                        if fresh > 0:
                            pos.current_price = fresh
                            pos.pnl_pct = round(
                                (fresh - pos.avg_price) / pos.avg_price * 100, 4
                            ) if pos.avg_price > 0 else 0.0
            except Exception:
                pass  # 실패 시 holdings lastPrice 그대로 사용

        positions = raw_positions
        mv = data.get("marketValue", {}).get("amount", {})
        total_krw = int(float(mv.get("krw", 0) or 0))
        usd_market_value = float(mv.get("usd", 0) or 0)
        # 예수금 (매수가능금액)
        cash = 0
        try:
            bp = requests.get(
                f"{self.BASE_URL}/api/v1/buying-power",
                headers=headers,
                params={"currency": "KRW"},
                timeout=10,
            )
            bp.raise_for_status()
            cash = int(float(bp.json().get("result", {}).get("cashBuyingPower", 0) or 0))
        except Exception:
            pass

        # 환율 (USD → KRW)
        usd_to_krw = 0.0
        try:
            fx = requests.get(
                f"{self.BASE_URL}/api/v1/exchange-rate",
                headers=self._headers(),
                params={"baseCurrency": "USD", "quoteCurrency": "KRW"},
                timeout=10,
            )
            fx.raise_for_status()
            usd_to_krw = float(fx.json().get("result", {}).get("rate", 0) or 0)
        except Exception:
            pass

        return Balance(
            cash=cash,
            total_value=total_krw + cash,
            positions=positions,
            usd_market_value=usd_market_value,
            usd_to_krw=usd_to_krw,
        )

    def get_ohlcv(self, symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
        """토스 /api/v1/candles 로 일봉 OHLCV 조회 (최대 200봉)"""
        candles = []
        before = None
        while True:
            params: dict = {"symbol": symbol, "interval": "1d", "count": 200, "adjusted": "true"}
            if before:
                params["before"] = before
            resp = requests.get(
                f"{self.BASE_URL}/api/v1/candles",
                headers=self._headers(),
                params=params,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})
            batch = result.get("candles", [])
            candles.extend(batch)
            before = result.get("nextBefore")
            if not before or not batch:
                break
            # start 날짜 이전 데이터까지만 수집
            if batch[-1]["timestamp"][:10] <= start:
                break

        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = []
        for c in candles:
            date = c["timestamp"][:10]
            if date < start:
                continue
            if end and date > end:
                continue
            rows.append({
                "date": date,
                "open": float(c["openPrice"]),
                "high": float(c["highPrice"]),
                "low": float(c["lowPrice"]),
                "close": float(c["closePrice"]),
                "volume": float(c["volume"]),
            })

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).sort_values("date")
        df.index = pd.to_datetime(df["date"])
        return df[["open", "high", "low", "close", "volume"]]

    def get_us_purchase_fx(self) -> dict:
        """
        US 주식 종목별 매수 당시 USD/KRW 가중평균 환율.
        /api/v1/orders?status=CLOSED 전체를 페이지네이션 수집 후
        FinanceDataReader USD/KRW 히스토리로 매수일 환율 계산.

        반환: {
            symbol: {wavg_fx, min_fx, max_fx, buys, total_usd, total_krw_at_purchase}
        }
        """
        from collections import defaultdict

        self._ensure_token()
        self._ensure_account_seq()

        all_orders: list[dict] = []
        cursor: Optional[str] = None
        while True:
            params: dict = {"status": "CLOSED"}
            if cursor:
                params["cursor"] = cursor
            # 429 rate-limit 대응: 0.3s 기본, 429 시 2s 재시도
            for attempt in range(3):
                time.sleep(0.3 if attempt == 0 else 2.0)
                resp = requests.get(
                    f"{self.BASE_URL}/api/v1/orders",
                    headers=self._headers(with_account=True),
                    params=params,
                    timeout=10,
                )
                if resp.status_code != 429:
                    break
            resp.raise_for_status()
            data = resp.json().get("result", {})
            all_orders.extend(data.get("orders", []))
            if not data.get("hasNext"):
                break
            cursor = data.get("nextCursor")

        # US BUY 주문 집계
        us_buys: dict[str, list[dict]] = defaultdict(list)
        all_dates: set[str] = set()
        for o in all_orders:
            sym = o.get("symbol", "")
            if not sym.strip().isdigit() and o.get("side") == "BUY":
                ex = o.get("execution") or {}
                filled_qty = float(ex.get("filledQuantity") or 0)
                filled_usd = float(ex.get("filledAmount") or 0)
                filled_at = (ex.get("filledAt") or "")[:10]
                if filled_qty > 0 and filled_at:
                    us_buys[sym].append({"date": filled_at, "qty": filled_qty, "usd": filled_usd})
                    all_dates.add(filled_at)

        if not all_dates:
            return {}

        # USD/KRW 히스토리 (매수일 범위 전체)
        min_year = min(all_dates)[:4]
        today_str = datetime.today().strftime("%Y-%m-%d")
        fx_df = fdr.DataReader("USD/KRW", f"{min_year}-01-01", today_str)
        fx_df.index = fx_df.index.strftime("%Y-%m-%d")
        fx_close = fx_df["Close"]

        def _get_fx(date_str: str) -> Optional[float]:
            if date_str in fx_close.index:
                return float(fx_close[date_str])
            avail = fx_close.index[fx_close.index <= date_str]
            return float(fx_close[avail[-1]]) if len(avail) > 0 else None

        result: dict[str, dict] = {}
        for sym, buys in us_buys.items():
            rates_w: list[tuple[float, float]] = []  # (usd_amount, fx_rate)
            total_krw = 0.0
            for b in buys:
                fx = _get_fx(b["date"])
                if fx:
                    total_krw += b["usd"] * fx
                    rates_w.append((b["usd"], fx))
            if rates_w:
                wavg = sum(r[0] * r[1] for r in rates_w) / sum(r[0] for r in rates_w)
                fxs = [r[1] for r in rates_w]
                result[sym] = {
                    "wavg_fx": round(wavg),
                    "min_fx": round(min(fxs)),
                    "max_fx": round(max(fxs)),
                    "buys": len(buys),
                    "total_usd": sum(b["usd"] for b in buys),
                    "total_krw_at_purchase": int(total_krw),
                }

        return result

    def place_order(self, symbol: str, side: str, quantity: int, price: int = 0) -> OrderResult:
        order_type = "MARKET" if price == 0 else "LIMIT"
        payload: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "orderType": order_type,
            "quantity": str(quantity),   # 스펙: 문자열 정수
        }
        if price > 0:
            payload["price"] = str(price)
        try:
            resp = requests.post(
                f"{self.BASE_URL}/api/v1/orders",
                headers=self._headers(with_account=True),
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json().get("result", {})
            return OrderResult(success=True, order_id=result.get("orderId", ""), message="주문 완료")
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
