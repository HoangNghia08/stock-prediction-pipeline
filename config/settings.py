"""
Cấu hình trung tâm cho toàn bộ pipeline. Mọi hằng số/giá trị mặc định nên
nằm ở đây thay vì rải rác trong code logic - giúp thay đổi cấu hình
(ví dụ đổi model FinBERT, đổi ngưỡng relevance) mà không cần sửa logic.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ---- Đường dẫn ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

for _dir in (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---- Cấu hình FinBERT ----
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 16

# ---- Cấu hình News Collector ----
GNEWS_MAX_RESULTS = 100
GNEWS_REQUEST_DELAY_SEC = 1.5
HEADLINE_MATCH_WEIGHT = 2
SNIPPET_MATCH_WEIGHT = 1
MAX_RELEVANCE_SCORE_CAP = 6
MIN_RELEVANCE_SCORE = 1

# Danh sách các trang tin tài chính uy tín để mở rộng phạm vi tìm kiếm,
# thay vì chỉ giới hạn 1 nguồn duy nhất (finance.yahoo.com). Có thể tự
# thêm/bớt nguồn tùy nhu cầu - càng nhiều nguồn, càng nhiều dữ liệu thô,
# nhưng cũng cần cân nhắc thời gian crawl tăng theo.
DEFAULT_NEWS_SOURCES: List[str] = [
    "finance.yahoo.com",
    "reuters.com",
    "cnbc.com",
    "marketwatch.com",
    "investing.com",
    "barrons.com",
    "seekingalpha.com",
    "fool.com",
]

# Chia nhỏ khoảng thời gian crawl theo tháng thay vì gọi 1 truy vấn duy
# nhất cho cả khoảng thời gian dài - vì Google News RSS giới hạn số kết
# quả trả về CHO MỖI TRUY VẤN (không phải cho mỗi khoảng ngày), nên chia
# nhỏ giúp lấy được nhiều dữ liệu hơn đáng kể.
NEWS_CHUNK_SIZE_DAYS = 30

# ---- Cấu hình Sentiment Decay ----
# Half-life (tính theo NGÀY GIAO DỊCH) cho việc suy giảm ảnh hưởng của tin
# tức theo thời gian - ví dụ half-life=3 nghĩa là sau 3 ngày giao dịch,
# ảnh hưởng còn lại của 1 tin tức chỉ còn ~50% so với ban đầu.
# Đây là siêu tham số cần tinh chỉnh thực nghiệm, không có giá trị "đúng"
# tuyệt đối - 3 là điểm khởi đầu hợp lý cho tin tức tài chính ngắn hạn.
SENTIMENT_DECAY_HALFLIFE_DAYS = 3.0


# ---- Alias theo ticker - MỞ RỘNG TẠI ĐÂY khi thêm mã cổ phiếu mới ----
DEFAULT_TICKER_ALIASES: Dict[str, List[str]] = {
    "TSLA": ["tesla", "tsla", "$tsla", "elon musk"],
}

# Giờ đóng cửa thị trường (ET) - dùng để xác định tin "sau giờ đóng cửa"
# cần dồn sang ngày giao dịch kế tiếp, tránh look-ahead bias.
MARKET_CLOSE_HOUR_ET = 16  # 4:00 PM ET
MARKET_TIMEZONE = "America/New_York"


@dataclass
class TickerConfig:
    """Cấu hình cho 1 mã cổ phiếu cụ thể trong 1 lần chạy pipeline."""

    ticker: str
    company_name: str
    start_date: str
    end_date: str
    extra_aliases: List[str] = field(default_factory=list)

    def all_aliases(self) -> List[str]:
        aliases = list(DEFAULT_TICKER_ALIASES.get(self.ticker.upper(), []))
        aliases += [self.ticker, self.company_name]
        aliases += self.extra_aliases
        return list(dict.fromkeys(a.lower() for a in aliases if a))
