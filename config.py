import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # 시세 조회용 증권사 (차트·분석·백테스팅)
    data_provider: str = field(default_factory=lambda: os.getenv("DATA_PROVIDER", "kis"))
    # 실제 주문용 증권사 (매수·매도·잔고)
    trade_provider: str = field(default_factory=lambda: os.getenv("TRADE_PROVIDER", "toss"))

    # KIS
    kis_app_key: str = field(default_factory=lambda: os.getenv("KIS_APP_KEY", ""))
    kis_app_secret: str = field(default_factory=lambda: os.getenv("KIS_APP_SECRET", ""))
    kis_account_no: str = field(default_factory=lambda: os.getenv("KIS_ACCOUNT_NO", ""))
    kis_is_paper: bool = field(default_factory=lambda: os.getenv("KIS_IS_PAPER", "true").lower() == "true")

    # 토스증권
    toss_client_id: str = field(default_factory=lambda: os.getenv("TOSS_CLIENT_ID", ""))
    toss_client_secret: str = field(default_factory=lambda: os.getenv("TOSS_CLIENT_SECRET", ""))
    toss_account_no: str = field(default_factory=lambda: os.getenv("TOSS_ACCOUNT_NO", ""))

    # 공통
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    discord_webhook_url: str = field(default_factory=lambda: os.getenv("DISCORD_WEBHOOK_URL", ""))

    # 매매 설정
    max_positions: int = field(default_factory=lambda: int(os.getenv("MAX_POSITIONS", "10")))
    position_size_pct: float = field(default_factory=lambda: float(os.getenv("POSITION_SIZE_PCT", "0.10")))
    max_daily_loss_pct: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")))
    stop_loss_pct: float = field(default_factory=lambda: float(os.getenv("STOP_LOSS_PCT", "0.07")))
    min_signal_score: float = field(default_factory=lambda: float(os.getenv("MIN_SIGNAL_SCORE", "0.55")))

    # Claude 모델
    claude_model: str = "claude-sonnet-4-6"           # 핵심 의사결정
    claude_report_model: str = "claude-sonnet-4-6"    # 전체 리포트 생성


config = Config()
