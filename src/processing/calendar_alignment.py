"""
Tổng hợp sentiment theo ngày + căn chỉnh về ngày giao dịch (trading day).

Mục đích của đoạn code này là chuẩn hóa ngày giao dịch của tin tức cho sát với thực tế,
đồng thời khắc phục rủi ro look-ahead bias từ bản gốc. Vì thị trường nghỉ cuối tuần và đóng cửa sau 4:00 PM ET hàng ngày,
mọi tin tức xuất hiện trong các khung giờ này sẽ được tự động chuyển sang ngày giao dịch tiếp theo.

Lưu ý: đây vẫn chỉ xử lý theo lịch dương (Sat/Sun), chưa xử lý ngày lễ
giao dịch NYSE.
"""

import pandas as pd

from config.settings import MARKET_CLOSE_HOUR_ET, MARKET_TIMEZONE
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def _assign_effective_trading_date(published_at_utc: pd.Series) -> pd.Series:
    """
    Xác định ngày giao dịch mà 1 tin tức thực sự có thể ảnh hưởng tới,
    dựa trên thời điểm đăng (so với giờ đóng cửa thị trường) và ngày trong tuần.
    """
    local_time = published_at_utc.dt.tz_convert(MARKET_TIMEZONE)
    effective_date = local_time.dt.normalize()

    # Tin đăng sau giờ đóng cửa (>= 16:00 ET) -> dồn sang ngày hôm sau
    after_close = local_time.dt.hour >= MARKET_CLOSE_HOUR_ET
    effective_date = effective_date.where(~after_close, effective_date + pd.Timedelta(days=1))

    # Nếu ngày hiệu lực rơi vào cuối tuần -> dồn tiếp sang thứ Hai
    weekday = effective_date.dt.weekday
    effective_date = effective_date.mask(weekday == 5, effective_date + pd.Timedelta(days=2))  # Sat
    weekday = effective_date.dt.weekday
    effective_date = effective_date.mask(weekday == 6, effective_date + pd.Timedelta(days=1))  # Sun

    return effective_date.dt.strftime("%Y-%m-%d")


def aggregate_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    Gộp tin tức về đúng ngày giao dịch có hiệu lực, rồi tổng hợp theo ngày
    với trọng số theo Relevance_Score (bài liên quan mạnh chi phối điểm
    trung bình nhiều hơn bài chỉ liên quan mờ nhạt).

    Trả về DataFrame với cột: Date, News_Count, Positive_Count,
    Negative_Count, Neutral_Count, Avg_Relevance_Score,
    Weighted_Sentiment_Score, Net_Sentiment_Score, Has_News.
    """
    if df_news.empty:
        return pd.DataFrame()

    df = df_news.copy()
    df["Date"] = _assign_effective_trading_date(df["Published_At_UTC"])

    def _weighted_signed_score(group: pd.DataFrame) -> float:
        total_weight = group["Relevance_Score"].sum()
        if total_weight == 0:
            return 0.0
        return (group["Signed_Sentiment_Score"] * group["Relevance_Score"]).sum() / total_weight

    daily = (
        df.groupby("Date")
        .apply(
            lambda g: pd.Series(
                {
                    "News_Count": len(g),
                    "Positive_Count": (g["Sentiment_Label"] == "Positive").sum(),
                    "Negative_Count": (g["Sentiment_Label"] == "Negative").sum(),
                    "Neutral_Count": (g["Sentiment_Label"] == "Neutral").sum(),
                    "Avg_Relevance_Score": g["Relevance_Score"].mean(),
                    "Weighted_Sentiment_Score": _weighted_signed_score(g),
                }
            )
        )
        .reset_index()
    )

    daily["Net_Sentiment_Score"] = (
        daily["Positive_Count"] - daily["Negative_Count"]
    ) / daily["News_Count"]

    # Cờ nhị phân phân biệt "không có tin tức" với "có tin nhưng trung tính" -
    # tránh mất thông tin khi điền 0 cho Weighted_Sentiment_Score ở bước merge.
    daily["Has_News"] = 1

    logger.info("Tổng hợp xong sentiment cho %d ngày giao dịch.", len(daily))
    return daily
