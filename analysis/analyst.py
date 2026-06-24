"""
Claude AI 기반 애널리스트.

비용 최적화 전략:
  1. should_invoke()로 신호가 약한 종목은 Claude 미호출 → 규칙 기반 즉시 판단
  2. System prompt 캐싱(cache_control="ephemeral")으로 입력 토큰 75% 절감
  3. 의사결정(max_tokens=600)과 리포트(max_tokens=2500)를 분리
  4. 호출 통계를 usage_stats로 실시간 모니터링
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from config import config
from indicators.technical import TechnicalSignals

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """당신은 15년 경력의 한국 주식 전문 퀀트 애널리스트입니다.
기술적 분석과 재무 지표를 통합하여 명확한 투자 판단을 제시합니다.

판단 기준:
- BUY : 기술적 강세 + 모멘텀 확인 + 재무 건전
- SELL: 기술적 약세 또는 손절 조건
- HOLD: 신호 혼재, 방향성 불명확
- SKIP: 데이터 부족 또는 극단적 위험

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 금지):
{
  "action": "BUY|SELL|HOLD|SKIP",
  "confidence": 0.0~1.0,
  "target_price": 정수_또는_null,
  "stop_loss": 정수_또는_null,
  "hold_period": "단기(1-2주)|중기(1-3개월)|장기(3개월+)|없음",
  "summary": "2-3문장 한국어 요약",
  "key_factors": ["요인1", "요인2", "요인3"]
}"""


class ClaudeAnalyst:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0

    # ── 사전 필터 (Claude 호출 여부 결정) ────────────────────────────────────

    def should_invoke(self, signals: TechnicalSignals) -> bool:
        """신호 강도가 임계값 이상이거나 특이 이벤트 발생 시에만 True"""
        score = abs(signals.composite_score)
        return any([
            score >= config.min_signal_score,
            signals.macd_crossover is not None,
            signals.rsi_signal in ("overbought", "oversold") and score > 0.35,
            signals.bb_squeeze,
            signals.volume_ratio > 2.5,
        ])

    # ── 규칙 기반 즉시 판단 (Claude 미호출) ──────────────────────────────────

    def quick_decision(self, signals: TechnicalSignals) -> dict:
        score = signals.composite_score
        if score >= 0.55:
            action, conf = "BUY", min(1.0, score + 0.1)
        elif score <= -0.45:
            action, conf = "SELL", min(1.0, abs(score) + 0.1)
        else:
            action, conf = "HOLD", round(1.0 - abs(score), 2)

        cross_note = ""
        if signals.macd_crossover == "golden":
            cross_note = " MACD 골든크로스 포착."
        elif signals.macd_crossover == "death":
            cross_note = " MACD 데드크로스 포착."

        return {
            "action": action,
            "confidence": round(conf, 2),
            "target_price": None,
            "stop_loss": None,
            "hold_period": "없음",
            "summary": (
                f"규칙 기반 자동 판단 (종합점수: {score:+.3f})."
                f" RSI {signals.rsi:.1f}({signals.rsi_signal}),"
                f" MA {'골든크로스' if signals.golden_cross else '데드크로스' if signals.golden_cross is False else '중립'}."
                + cross_note
            ),
            "key_factors": [
                f"RSI {signals.rsi:.1f} → {signals.rsi_signal}",
                f"MACD 히스토그램 {'양봉' if signals.macd_hist > 0 else '음봉'} ({signals.macd_hist:+.4f})",
                f"볼린저 위치 {signals.bb_position*100:.0f}% / 거래량 {signals.volume_ratio:.1f}x",
            ],
            "claude_used": False,
        }

    # ── 핵심 의사결정 (짧은 응답, 비용 최소화) ───────────────────────────────

    def decide(
        self,
        symbol: str,
        name: str,
        price: int,
        signals: TechnicalSignals,
        fundamental: Optional[dict] = None,
    ) -> dict:
        if not self.should_invoke(signals):
            return self.quick_decision(signals)

        prompt = self._decision_prompt(symbol, name, price, signals, fundamental)
        try:
            resp = self.client.messages.create(
                model=config.claude_model,
                max_tokens=600,
                system=[{"type": "text", "text": _SYSTEM_PROMPT,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
            )
            self._track(resp.usage)
            text = resp.content[0].text
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                result = json.loads(m.group())
                result["claude_used"] = True
                return result
        except Exception as e:
            logger.warning("Claude decide failed: %s", e)

        return self.quick_decision(signals)

    # ── 전체 애널리스트 리포트 (온디맨드) ────────────────────────────────────

    def generate_report(
        self,
        symbol: str,
        name: str,
        price: int,
        signals: TechnicalSignals,
        fundamental: Optional[dict] = None,
    ) -> str:
        fund_str = self._fmt_fundamental(fundamental)
        prompt = f"""[{name} ({symbol})] 종합 투자 리포트를 작성해주세요.

== 시장 데이터 ==
현재가: {price:,}원

== 기술적 지표 ==
RSI(14): {signals.rsi:.1f} → {signals.rsi_signal}
MACD: 라인={signals.macd:+.4f} / 시그널={signals.macd_signal_line:+.4f} / 히스토={signals.macd_hist:+.4f}{" ★골든크로스!" if signals.macd_crossover=="golden" else " ★데드크로스!" if signals.macd_crossover=="death" else ""}
이동평균: MA20={signals.sma_20:,.0f} / MA60={signals.sma_60:,.0f} / MA120={signals.sma_120:,.0f}
MA추세: {signals.ma_trend} / 현재가 MA20대비: {signals.price_vs_ma20_pct:+.2f}%
{"★ 골든크로스 (MA20 > MA60)" if signals.golden_cross else "★ 데드크로스 (MA20 < MA60)" if signals.golden_cross is False else ""}
볼린저밴드: 상={signals.bb_upper:,.0f} / 중={signals.bb_middle:,.0f} / 하={signals.bb_lower:,.0f} (위치 {signals.bb_position*100:.0f}%){" ⚡스퀴즈-변동성 폭발 임박" if signals.bb_squeeze else ""}
스토캐스틱: K={signals.stoch_k:.1f} / D={signals.stoch_d:.1f}
거래량: 20일 평균의 {signals.volume_ratio:.1f}배
종합점수: {signals.composite_score:+.3f}  (-1=강매도 ~ +1=강매수)

{fund_str}

다음 형식으로 작성:
## 투자의견: [BUY/HOLD/SELL]  목표주가: X,XXX원  손절: X,XXX원

### 핵심 투자 포인트
1. ...
2. ...
3. ...

### 기술적 분석 상세
...

### 리스크 요인
...

### 결론 및 매매 전략
..."""

        resp = self.client.messages.create(
            model=config.claude_report_model,
            max_tokens=2500,
            system=[{"type": "text", "text": _SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        self._track(resp.usage)
        return resp.content[0].text

    # ── 헬퍼 ─────────────────────────────────────────────────────────────────

    def _decision_prompt(self, symbol, name, price, signals, fundamental) -> str:
        fund_str = self._fmt_fundamental(fundamental)
        cross = (" [골든크로스]" if signals.macd_crossover == "golden"
                 else " [데드크로스]" if signals.macd_crossover == "death" else "")
        return (
            f"{name}({symbol}) 현재가={price:,}원\n"
            f"RSI={signals.rsi:.1f}({signals.rsi_signal}), "
            f"MACD히스={signals.macd_hist:+.4f}{cross}\n"
            f"MA={'골든크로스' if signals.golden_cross else '데드크로스' if signals.golden_cross is False else '중립'}"
            f" 추세={signals.ma_trend}\n"
            f"BB위치={signals.bb_position*100:.0f}%{'(스퀴즈)' if signals.bb_squeeze else ''}"
            f" 거래량={signals.volume_ratio:.1f}x\n"
            f"스토캐스틱 K={signals.stoch_k:.1f}/D={signals.stoch_d:.1f}"
            f" 종합={signals.composite_score:+.3f}\n"
            f"{fund_str}\nJSON으로 판단:"
        )

    def _fmt_fundamental(self, fundamental: Optional[dict]) -> str:
        if not fundamental:
            return ""
        return (
            f"재무: PER={fundamental.get('per','N/A')}, PBR={fundamental.get('pbr','N/A')}, "
            f"ROE={fundamental.get('roe','N/A')}%, 부채율={fundamental.get('debt_ratio','N/A')}%, "
            f"시총={fundamental.get('market_cap_b','N/A')}억원"
        )

    def _track(self, usage):
        self._calls += 1
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens

    @property
    def usage_stats(self) -> dict:
        # Sonnet 4.6 기준 캐시 히트 시 입력 0.30$/M, 출력 3.75$/M (캐시 미스 3$/M)
        cost = self._input_tokens * 3e-6 + self._output_tokens * 15e-6
        return {
            "calls": self._calls,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "est_cost_usd": round(cost, 4),
            "est_cost_krw": round(cost * 1380, 0),
        }
