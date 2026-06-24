"""
AI 자동 매매 시스템 대시보드
실행: streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
import os
from datetime import date, timedelta
from pathlib import Path

# 패키지 루트를 sys.path에 추가
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import FinanceDataReader as fdr

from config import config
from data.provider import create_data_provider, create_trade_provider
from data.names import get_name, get_name_with_code, search_krx
from indicators.technical import calculate_signals
from analysis.analyst import ClaudeAnalyst
from decision.engine import DecisionEngine
from execution.orders import OrderManager
from backtesting.engine import BacktestEngine, BUILT_IN_STRATEGIES

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI 자동 매매",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 캐시된 공유 객체 ─────────────────────────────────────────────────────────

@st.cache_resource
def get_components():
    data_prov  = create_data_provider(config)   # 시세·차트·분석
    trade_prov = create_trade_provider(config)  # 주문·잔고
    analyst    = ClaudeAnalyst()
    engine     = DecisionEngine(data_prov, analyst)
    order_mgr  = OrderManager(trade_prov, paper_trading=False)
    return data_prov, trade_prov, analyst, engine, order_mgr


data_prov, trade_prov, analyst, engine, order_mgr = get_components()
provider = data_prov  # 기존 코드 호환용

# ─── 차트 생성 헬퍼 (탭보다 먼저 정의) ──────────────────────────────────────

def _make_chart(df: pd.DataFrame, sigs, name: str) -> go.Figure:
    """캔들스틱 + MA + 볼린저 + 거래량 + RSI + MACD 4단 차트"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.15, 0.17, 0.18],
        vertical_spacing=0.02,
        subplot_titles=("가격 / 이동평균 / 볼린저", "거래량", "RSI(14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="캔들", increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a",
    ), row=1, col=1)

    c = df["close"]
    for window, color, label in [(20, "#ffd700", "MA20"), (60, "#1e90ff", "MA60"), (120, "#ff69b4", "MA120")]:
        ma = c.rolling(window).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=ma, name=label,
            line=dict(color=color, width=1.2), opacity=0.9,
        ), row=1, col=1)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_lo = bb_mid - 2 * bb_std
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_up, name="BB상단",
        line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_lo, name="BB하단",
        line=dict(color="rgba(150,150,255,0.5)", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(150,150,255,0.05)",
    ), row=1, col=1)

    vol_colors = [
        "#ef5350" if float(df["close"].iloc[i]) >= float(df["open"].iloc[i]) else "#26a69a"
        for i in range(len(df))
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="거래량",
        marker_color=vol_colors, opacity=0.7,
    ), row=2, col=1)

    delta = c.diff()
    avg_g = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_l = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = avg_g / avg_l.replace(0, 1e-10)
    rsi_s = 100 - 100 / (1 + rs)
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi_s, name="RSI",
        line=dict(color="#a78bfa", width=1.5),
    ), row=3, col=1)
    for level in [70, 30]:
        fig.add_hline(y=level, line_dash="dot",
                      line_color="rgba(255,255,255,0.3)", row=3, col=1)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_l = ema12 - ema26
    macd_sig = macd_l.ewm(span=9, adjust=False).mean()
    macd_hist_vals = macd_l - macd_sig
    hist_colors = ["#ef5350" if v >= 0 else "#26a69a" for v in macd_hist_vals]
    fig.add_trace(go.Bar(
        x=df.index, y=macd_hist_vals, name="히스토그램",
        marker_color=hist_colors, opacity=0.8,
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_l, name="MACD",
        line=dict(color="#00d4ff", width=1.5),
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_sig, name="시그널",
        line=dict(color="#ff7f0e", width=1.5),
    ), row=4, col=1)

    fig.update_layout(
        template="plotly_dark", title=name, height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    return fig


# ─── 사이드바 ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 AI 자동 매매")
    data_label  = "토스증권" if config.data_provider == "toss" else "한국투자증권"
    trade_label = "토스증권" if config.trade_provider == "toss" else "한국투자증권"
    st.info(f"📊 데이터: {data_label}")
    st.info(f"💳 주문: {trade_label}")
    st.divider()
    st.caption(f"Min Signal Score: {config.min_signal_score}")
    st.caption(f"포지션 크기: {config.position_size_pct*100:.0f}%")
    st.caption(f"손절: {config.stop_loss_pct*100:.0f}%")
    st.caption(f"최대 포지션: {config.max_positions}개")

# ─── 탭 ──────────────────────────────────────────────────────────────────────

tab_dash, tab_analysis, tab_backtest = st.tabs(
    ["📊 대시보드", "🔍 종목 분석", "📈 백테스팅"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: 대시보드
# ══════════════════════════════════════════════════════════════════════════════

with tab_dash:
    st.header("포트폴리오 현황")

    col_refresh, col_order = st.columns([1, 4])
    with col_refresh:
        if st.button("🔄 새로고침"):
            st.cache_data.clear()

    # 잔고 조회
    balance_placeholder = st.empty()
    try:
        balance = provider.get_balance()

        # 요약 카드
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("총 평가금액", f"{balance.total_value:,}원")
        c2.metric("예수금(현금)", f"{balance.cash:,}원")
        c3.metric("보유 종목 수", f"{len(balance.positions)}개")
        invested = balance.total_value - balance.cash
        c4.metric("투자 금액", f"{invested:,}원")

        # 포지션 테이블
        if balance.positions:
            st.subheader("보유 종목")
            rows = []
            for p in balance.positions:
                pnl_color = "🟢" if p.pnl_pct >= 0 else "🔴"
                rows.append({
                    "종목코드": p.symbol,
                    "종목명": p.name,
                    "보유수량": p.quantity,
                    "평균단가": f"{p.avg_price:,.0f}",
                    "현재가": f"{p.current_price:,}",
                    "수익률": f"{pnl_color} {p.pnl_pct:+.2f}%",
                    "평가금액": f"{p.current_price * p.quantity:,}원",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # 손절 경고
            stop_triggers = engine.check_stop_losses([
                {"symbol": p.symbol, "name": p.name,
                 "quantity": p.quantity, "avg_price": p.avg_price,
                 "current_price": p.current_price}
                for p in balance.positions
            ])
            if stop_triggers:
                st.error(f"⚠️ 손절 발동: {', '.join(t['symbol'] for t in stop_triggers)}")
                for t in stop_triggers:
                    if st.button(f"🔴 {t['symbol']} 즉시 손절", key=f"sl_{t['symbol']}"):
                        from decision.engine import TradeDecision
                        from indicators.technical import TechnicalSignals
                        # 최소한의 decision 객체로 매도 실행
                        d = TradeDecision(
                            symbol=t["symbol"], name=t["name"],
                            action="SELL", confidence=1.0,
                            price=t["current_price"], quantity=t["quantity"],
                            target_price=None, stop_loss=None,
                            hold_period="없음", summary=t["reason"],
                            key_factors=[], signals=None, claude_used=False,
                        )
                        rec = order_mgr.execute(d)
                        st.success(f"매도 주문: {rec.order_id}") if rec and rec.status == "success" else st.error("주문 실패")
        else:
            st.info("보유 종목 없음")

    except Exception as e:
        st.warning(f"잔고 조회 실패: {e}\n\n.env 파일의 API 키를 확인하세요.")

    # 수동 주문
    st.divider()
    st.subheader("수동 주문")
    col_sym, col_mname, col_side, col_qty, col_btn = st.columns([2, 2, 1, 1, 1])
    with col_sym:
        manual_symbol = st.text_input("종목코드", placeholder="005930")
    with col_mname:
        mname = get_name_with_code(manual_symbol.strip()) if manual_symbol else ""
        st.text_input("종목명", value=mname, disabled=True, key="manual_name")
    with col_side:
        manual_side = st.selectbox("구분", ["BUY", "SELL"])
    with col_qty:
        manual_qty = st.number_input("수량", min_value=1, value=1, step=1)
    with col_btn:
        st.write("")
        st.write("")
        if st.button("주문 실행", type="primary"):
            if manual_symbol:
                try:
                    price_info = provider.get_price(manual_symbol)
                    from decision.engine import TradeDecision
                    d = TradeDecision(
                        symbol=manual_symbol, name=price_info.name,
                        action=manual_side, confidence=1.0,
                        price=price_info.price, quantity=int(manual_qty),
                        target_price=None, stop_loss=None,
                        hold_period="없음", summary="수동 주문",
                        key_factors=[], signals=None, claude_used=False,
                    )
                    rec = order_mgr.execute(d)
                    if rec and rec.status == "success":
                        st.success(f"✅ 주문 완료: {rec.order_id}")
                    else:
                        st.error("주문 실패")
                except Exception as e:
                    st.error(f"오류: {e}")

    # 주문 이력
    st.divider()
    st.subheader("최근 주문 이력")
    history = order_mgr.get_history(50)
    if history:
        rows = [
            {
                "시간": r.timestamp[:19],
                "종목": f"{r.name or get_name(r.symbol)} ({r.symbol})",
                "구분": "🟢 매수" if r.side == "BUY" else "🔴 매도",
                "수량": r.quantity,
                "가격": f"{r.price:,}원",
                "상태": "✅" if r.status == "success" else "❌",
                "AI사용": "🤖" if r.claude_used else "📐",
                "사유": r.reason[:50],
            }
            for r in history
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("주문 이력 없음")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: 종목 분석
# ══════════════════════════════════════════════════════════════════════════════

with tab_analysis:
    st.header("AI 종목 분석")

    col_inp, col_name, col_date = st.columns([2, 2, 2])
    with col_inp:
        symbol_input = st.text_input(
            "종목코드 / 티커", placeholder="005930  또는  AAPL", key="analysis_symbol"
        )
    with col_name:
        # 코드 입력 즉시 종목명 표시
        if symbol_input:
            resolved = get_name_with_code(symbol_input.strip())
            st.text_input("종목명", value=resolved, disabled=True, key="analysis_name")
        else:
            st.text_input("종목명", value="", disabled=True,
                          placeholder="코드 입력 시 자동 표시", key="analysis_name")
    with col_date:
        chart_start = st.date_input(
            "차트 시작일", value=date.today() - timedelta(days=365), key="analysis_start"
        )

    # 종목 검색 (이름으로 찾기)
    with st.expander("🔎 종목 검색 (이름으로 찾기)"):
        search_q = st.text_input("종목명 검색", placeholder="삼성, 카카오, NAVER ...", key="search_q")
        if search_q:
            results = search_krx(search_q, limit=8)
            if results:
                labels = [r["label"] for r in results]
                chosen = st.radio("종목 선택", labels, key="search_result", horizontal=True)
                if chosen:
                    # 선택 시 코드 추출해서 안내
                    chosen_code = chosen.split("(")[-1].rstrip(")")
                    st.info(f"위 '종목코드' 입력란에 **{chosen_code}** 를 입력하세요.")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        do_analyze = st.button("🔍 분석", type="primary")
    with col_btn2:
        do_report = st.button("📄 전체 리포트 생성")

    if symbol_input and (do_analyze or do_report):
        with st.spinner("데이터 로드 중..."):
            try:
                price_info = provider.get_price(symbol_input)
                df = provider.get_ohlcv(symbol_input, chart_start.strftime("%Y-%m-%d"))

                if len(df) < 30:
                    st.error("데이터 부족 (최소 30일 필요)")
                    st.stop()

                sigs = calculate_signals(df, symbol_input)
                fundamental = {
                    "per": price_info.per, "pbr": price_info.pbr,
                    "market_cap_b": price_info.market_cap // 100_000_000,
                }

                # ── 가격 요약 ────────────────────────────────────────────────
                c1, c2, c3, c4, c5 = st.columns(5)
                chg_color = "normal" if price_info.change_pct >= 0 else "inverse"
                c1.metric("현재가", f"{price_info.price:,}원",
                          f"{price_info.change_pct:+.2f}%", delta_color=chg_color)
                c2.metric("RSI(14)", f"{sigs.rsi:.1f}", sigs.rsi_signal)
                c3.metric("MACD 히스토", f"{sigs.macd_hist:+.4f}",
                          sigs.macd_crossover or "—")
                c4.metric("종합 점수", f"{sigs.composite_score:+.3f}",
                          sigs.ma_trend)
                c5.metric("거래량 비율", f"{sigs.volume_ratio:.1f}x",
                          "⚡스퀴즈" if sigs.bb_squeeze else "")

                # ── 차트 ─────────────────────────────────────────────────────
                st.subheader(f"{price_info.name} ({symbol_input}) 기술적 차트")
                fig = _make_chart(df, sigs, price_info.name)
                st.plotly_chart(fig, use_container_width=True)

                # ── 지표 상세 ────────────────────────────────────────────────
                with st.expander("지표 상세 보기"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.write("**이동평균**")
                        st.write(f"MA20: {sigs.sma_20:,.0f}")
                        st.write(f"MA60: {sigs.sma_60:,.0f}")
                        st.write(f"MA120: {sigs.sma_120:,.0f}")
                        cross = "🟢 골든크로스" if sigs.golden_cross else "🔴 데드크로스" if sigs.golden_cross is False else "—"
                        st.write(f"MA상태: {cross}")
                    with col_b:
                        st.write("**볼린저밴드**")
                        st.write(f"상단: {sigs.bb_upper:,.0f}")
                        st.write(f"중단: {sigs.bb_middle:,.0f}")
                        st.write(f"하단: {sigs.bb_lower:,.0f}")
                        st.write(f"위치: {sigs.bb_position*100:.1f}%")
                    with col_c:
                        st.write("**기타**")
                        st.write(f"스토캐스틱 K: {sigs.stoch_k:.1f}")
                        st.write(f"스토캐스틱 D: {sigs.stoch_d:.1f}")
                        st.write(f"PER: {price_info.per:.1f}")
                        st.write(f"PBR: {price_info.pbr:.2f}")

                # ── AI 분석 ──────────────────────────────────────────────────
                if do_report:
                    st.subheader("📄 AI 애널리스트 리포트")
                    with st.spinner("Claude가 리포트를 작성 중..."):
                        report = analyst.generate_report(
                            symbol_input, price_info.name, price_info.price,
                            sigs, fundamental
                        )
                    st.markdown(report)
                elif do_analyze:
                    st.subheader("🤖 AI 투자 의견")
                    with st.spinner("Claude가 분석 중..."):
                        decision = analyst.decide(
                            symbol_input, price_info.name, price_info.price,
                            sigs, fundamental
                        )

                    action = decision.get("action", "HOLD")
                    action_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "SKIP": "⚫"}
                    conf = decision.get("confidence", 0)
                    ai_badge = "🤖 Claude" if decision.get("claude_used") else "📐 규칙기반"

                    col_act, col_conf, col_mode = st.columns(3)
                    col_act.metric(
                        "투자의견",
                        f"{action_colors.get(action, '')} {action}",
                    )
                    col_conf.metric("신뢰도", f"{conf*100:.0f}%")
                    col_mode.metric("판단 방식", ai_badge)

                    if decision.get("target_price"):
                        c1, c2 = st.columns(2)
                        c1.metric("목표주가", f"{decision['target_price']:,}원",
                                  f"{(decision['target_price']/price_info.price-1)*100:+.1f}%")
                        if decision.get("stop_loss"):
                            c2.metric("손절가", f"{decision['stop_loss']:,}원",
                                      f"{(decision['stop_loss']/price_info.price-1)*100:+.1f}%",
                                      delta_color="inverse")

                    st.info(decision.get("summary", ""))
                    if decision.get("key_factors"):
                        st.write("**핵심 요인**")
                        for f in decision["key_factors"]:
                            st.write(f"• {f}")

                    # 매수 실행 버튼
                    if action == "BUY":
                        try:
                            bal = provider.get_balance()
                            pos_value = bal.total_value * config.position_size_pct
                            qty = max(1, int(pos_value / price_info.price))
                            if st.button(f"🟢 {qty}주 매수 실행", type="primary"):
                                from decision.engine import TradeDecision
                                d = TradeDecision(
                                    symbol=symbol_input, name=price_info.name,
                                    action="BUY", confidence=conf,
                                    price=price_info.price, quantity=qty,
                                    target_price=decision.get("target_price"),
                                    stop_loss=decision.get("stop_loss"),
                                    hold_period=decision.get("hold_period", "없음"),
                                    summary=decision.get("summary", ""),
                                    key_factors=decision.get("key_factors", []),
                                    signals=sigs,
                                    claude_used=decision.get("claude_used", False),
                                )
                                rec = order_mgr.execute(d)
                                if rec and rec.status == "success":
                                    st.success(f"✅ 매수 완료! 주문번호: {rec.order_id}")
                                else:
                                    st.error("주문 실패")
                        except Exception:
                            pass

                # Claude 비용 표시
                stats = analyst.usage_stats
                st.caption(
                    f"Claude 사용: {stats['calls']}회 / "
                    f"{stats['input_tokens']:,} 입력토큰 / "
                    f"예상비용 ₩{stats['est_cost_krw']:,.0f}"
                )

            except Exception as e:
                st.error(f"오류: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: 백테스팅
# ══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.header("전략 백테스팅")

    # 설정 패널
    with st.expander("⚙️ 백테스트 설정", expanded=True):
        col_strat, col_sym = st.columns([1, 2])
        with col_strat:
            strategy_name = st.selectbox("전략", list(BUILT_IN_STRATEGIES.keys()))
        with col_sym:
            symbols_input = st.text_area(
                "종목코드 (줄바꿈 또는 쉼표 구분)",
                value="005930\n000660\n035720\n005380\n051910",
                height=120,
            )

        col_d1, col_d2, col_cap = st.columns(3)
        with col_d1:
            bt_start = st.date_input("시작일", value=date(2022, 1, 1), key="bt_start")
        with col_d2:
            bt_end = st.date_input("종료일", value=date.today(), key="bt_end")
        with col_cap:
            bt_capital = st.number_input("초기 자본 (원)", value=10_000_000, step=1_000_000,
                                         min_value=1_000_000)

        col_pos, col_sl, col_maxp = st.columns(3)
        with col_pos:
            bt_pos_size = st.slider("포지션 크기 (%)", 5, 30, 10) / 100
        with col_sl:
            bt_stop_loss = st.slider("손절 (%)", 3, 20, 7) / 100
        with col_maxp:
            bt_max_pos = st.slider("최대 포지션 수", 1, 20, 10)

    if st.button("▶ 백테스트 실행", type="primary"):
        syms = [
            s.strip() for s in symbols_input.replace(",", "\n").splitlines()
            if s.strip()
        ]
        if not syms:
            st.error("종목코드를 입력하세요")
        else:
            with st.spinner(f"{len(syms)}개 종목 × {(bt_end - bt_start).days}일 시뮬레이션 중..."):
                bt_engine = BacktestEngine(initial_capital=float(bt_capital))
                result = bt_engine.run(
                    symbols=syms,
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=bt_end.strftime("%Y-%m-%d"),
                    strategy_name=strategy_name,
                    position_size_pct=bt_pos_size,
                    stop_loss_pct=bt_stop_loss,
                    max_positions=bt_max_pos,
                )

            # ── 성과 지표 ─────────────────────────────────────────────────
            st.subheader("📊 성과 요약")
            ret_color = "normal" if result.total_return_pct >= 0 else "inverse"

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("총 수익률", f"{result.total_return_pct:+.2f}%",
                      f"KOSPI {result.benchmark_return_pct:+.2f}%",
                      delta_color=ret_color)
            c2.metric("연환산 수익률", f"{result.annual_return_pct:+.2f}%")
            c3.metric("최대 낙폭(MDD)", f"-{result.max_drawdown_pct:.2f}%",
                      delta_color="inverse")
            c4.metric("샤프 비율", f"{result.sharpe_ratio:.3f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("승률", f"{result.win_rate_pct:.1f}%")
            c6.metric("손익비", f"{result.profit_factor:.2f}")
            c7.metric("총 거래 수", f"{result.total_trades}회")
            c8.metric("소르티노 비율", f"{result.sortino_ratio:.3f}")

            # ── 수익 곡선 ─────────────────────────────────────────────────
            st.subheader("📈 수익 곡선 vs KOSPI")
            try:
                kospi_df = fdr.DataReader("KS11", bt_start.strftime("%Y-%m-%d"),
                                          bt_end.strftime("%Y-%m-%d"))
                kospi_norm = kospi_df["Close"] / kospi_df["Close"].iloc[0] * float(bt_capital)
            except Exception:
                kospi_norm = None

            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                name="전략", line=dict(color="#00d4ff", width=2),
            ))
            if kospi_norm is not None:
                fig_equity.add_trace(go.Scatter(
                    x=kospi_norm.index, y=kospi_norm.values,
                    name="KOSPI", line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                ))
            fig_equity.update_layout(
                template="plotly_dark", height=400,
                xaxis_title="날짜", yaxis_title="포트폴리오 가치 (원)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # ── 낙폭 차트 ─────────────────────────────────────────────────
            st.subheader("📉 낙폭(Drawdown)")
            eq = result.equity_curve
            dd = (eq - eq.cummax()) / eq.cummax() * 100
            fig_dd = go.Figure(go.Scatter(
                x=dd.index, y=dd.values,
                fill="tozeroy", name="낙폭 (%)",
                line=dict(color="#ff4444"), fillcolor="rgba(255,68,68,0.3)",
            ))
            fig_dd.update_layout(
                template="plotly_dark", height=200,
                xaxis_title="날짜", yaxis_title="낙폭 (%)",
                margin=dict(l=40, r=20, t=10, b=40),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── 거래 내역 ─────────────────────────────────────────────────
            st.subheader("📋 거래 내역")
            buy_trades = [t for t in result.trades if t.action == "BUY"]
            sell_trades = [t for t in result.trades if t.action == "SELL"]
            tab_buy, tab_sell = st.tabs([f"매수 ({len(buy_trades)}건)", f"매도 ({len(sell_trades)}건)"])

            with tab_buy:
                if buy_trades:
                    st.dataframe(pd.DataFrame([{
                        "날짜": t.date,
                        "종목": f"{get_name(t.symbol)} ({t.symbol})",
                        "수량": t.quantity, "가격": f"{t.price:,.0f}",
                        "금액": f"{t.value:,.0f}", "사유": t.reason,
                    } for t in buy_trades]), use_container_width=True, hide_index=True)

            with tab_sell:
                if sell_trades:
                    rows = []
                    for t in sell_trades:
                        color = "🟢" if t.pnl >= 0 else "🔴"
                        rows.append({
                            "날짜": t.date,
                            "종목": f"{get_name(t.symbol)} ({t.symbol})",
                            "수량": t.quantity, "가격": f"{t.price:,.0f}",
                            "손익": f"{color} {t.pnl:+,.0f}원 ({t.pnl_pct:+.2f}%)",
                            "사유": t.reason,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.success(
                f"✅ 백테스트 완료: {result.strategy_name} | "
                f"{bt_capital:,}원 → {result.final_capital:,.0f}원 "
                f"({result.total_return_pct:+.2f}%)"
            )
