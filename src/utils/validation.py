"""
Kiểm tra chất lượng dữ liệu (data validation) - bắt buộc trong ML production
để phát hiện dữ liệu bẩn trước khi nó lọt vào quá trình huấn luyện mô hình.
"""

import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raise khi dữ liệu không đạt các ràng buộc tối thiểu để dùng cho huấn luyện."""


def validate_price_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("DataFrame giá rỗng - không thể tiếp tục pipeline.")

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(f"Thiếu cột bắt buộc trong dữ liệu giá: {missing}")

    if (df["Close"] <= 0).any():
        raise DataValidationError("Phát hiện giá Close <= 0 - dữ liệu giá bất thường.")

    if df["Date"].duplicated().any():
        n_dup = df["Date"].duplicated().sum()
        raise DataValidationError(f"Phát hiện {n_dup} ngày bị trùng lặp trong dữ liệu giá.")

    dates = pd.to_datetime(df["Date"])
    if not dates.is_monotonic_increasing:
        raise DataValidationError("Cột Date không tăng dần đơn điệu - dữ liệu có thể bị xáo trộn.")

    logger.info("Validation dữ liệu giá: OK (%d dòng).", len(df))


def validate_merged_dataframe(df: pd.DataFrame, n_before_dropna: int) -> None:
    n_after = len(df)
    n_dropped = n_before_dropna - n_after
    drop_ratio = n_dropped / n_before_dropna if n_before_dropna > 0 else 0.0

    logger.info(
        "Đã xóa %d/%d dòng (%.2f%%) do NaN sau khi merge giá + sentiment.",
        n_dropped, n_before_dropna, drop_ratio * 100,
    )

    if drop_ratio > 0.05:
        logger.warning(
            "Tỷ lệ dòng bị xóa do NaN (%.2f%%) vượt ngưỡng cảnh báo 5%% - "
            "kiểm tra lại xem bước crawl tin tức hoặc merge có vấn đề không.",
            drop_ratio * 100,
        )

    sentiment_cols = [c for c in df.columns if "Sentiment" in c]
    for col in sentiment_cols:
        out_of_range = df[(df[col] < -1.01) | (df[col] > 1.01)]
        if not out_of_range.empty:
            logger.warning(
                "Cột %s có %d giá trị nằm ngoài khoảng [-1, 1] hợp lệ.",
                col, len(out_of_range),
            )
