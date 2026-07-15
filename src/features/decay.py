"""
Sentiment Decay - mô phỏng hiệu ứng ảnh hưởng của tin tức suy giảm dần
theo thời gian (thay vì biến mất hoàn toàn ngay ngày hôm sau).

Ý tưởng: dùng Exponentially Weighted Moving Average (EWMA) trên chuỗi
Weighted_Sentiment_Score theo NGÀY GIAO DỊCH (đã merge đầy đủ, có 0 cho
ngày không có tin). Công thức đệ quy:

    Decayed(t) = alpha * Raw(t) + (1 - alpha) * Decayed(t-1)

alpha được suy ra từ half-life (số ngày để ảnh hưởng còn lại ~50%).

QUAN TRỌNG: hàm này PHẢI được gọi SAU bước merge giá + sentiment và SAU
khi đã fillna(0) cho các ngày không có tin - nếu tính trên bảng tin tức
thô (chỉ có các ngày có tin), những ngày trống sẽ bị "nhảy cóc", làm sai
lệch hoàn toàn tốc độ suy giảm thực tế theo đúng lịch giao dịch.

Về tính nhân quả: EWMA tại thời điểm t chỉ phụ thuộc vào dữ liệu tại thời điểm t và
các thời điểm trước t (do công thức đệ quy lùi), nên không có rủi ro
look-ahead bias, miễn là DataFrame được sắp xếp tăng dần theo Date trước
khi gọi hàm này (pipeline hiện tại đã đảm bảo điều này qua price_collector
và calendar_alignment).
"""

import numpy as np
import pandas as pd

from config.settings import SENTIMENT_DECAY_HALFLIFE_DAYS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def halflife_to_alpha(halflife_days: float) -> float:
    """Chuyển half-life (số ngày) sang hệ số alpha dùng trong công thức EWMA đệ quy."""
    return 1 - np.exp(np.log(0.5) / halflife_days)


def apply_sentiment_decay(
    df_merged: pd.DataFrame,
    source_col: str = "Weighted_Sentiment_Score",
    output_col: str = "Decayed_Sentiment_Score",
    halflife_days: float = SENTIMENT_DECAY_HALFLIFE_DAYS,
) -> pd.DataFrame:
    """
    Thêm cột sentiment đã suy giảm theo thời gian (Decayed_Sentiment_Score)
    vào df_merged, dựa trên cột sentiment thô theo ngày (đã fillna 0).

    QUAN TRỌNG: df_merged phải đã được sắp xếp tăng dần theo Date trước
    khi gọi hàm này (bắt buộc để EWMA tính đúng theo đúng thứ tự thời gian).

    Giữ lại CẢ 2 cột (thô và đã decay) - không thay thế cột gốc, để mô
    hình downstream có thể tự học cách kết hợp cả 2 tín hiệu (tin hôm nay
    vs. dư âm tích lũy từ các ngày trước) thay vì chỉ có 1 lựa chọn duy nhất.
    """
    if df_merged.empty or source_col not in df_merged.columns:
        logger.warning(
            "Không thể tính sentiment decay: DataFrame rỗng hoặc thiếu cột '%s'.", source_col
        )
        return df_merged

    df = df_merged.copy()

    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        if not dates.is_monotonic_increasing:
            logger.warning(
                "DataFrame chưa được sắp xếp tăng dần theo Date - sắp xếp lại trước khi "
                "tính decay để đảm bảo tính nhân quả (tránh look-ahead bias)."
            )
            df = df.sort_values("Date").reset_index(drop=True)

    alpha = halflife_to_alpha(halflife_days)
    df[output_col] = df[source_col].ewm(alpha=alpha, adjust=False).mean()

    logger.info(
        "Đã tính '%s' từ '%s' với half-life=%.1f ngày (alpha=%.3f).",
        output_col, source_col, halflife_days, alpha,
    )
    return df
