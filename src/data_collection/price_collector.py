"""
Thu thập dữ liệu giá cổ phiếu qua yfinance.
"""

import pandas as pd
import yfinance as yf

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def collect_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Tải dữ liệu giá lịch sử cho đúng `ticker` trong khoảng [start_date, end_date].
    Trả về DataFrame với cột: Date, Open, High, Low, Close, Volume
    """
    logger.info("Đang tải dữ liệu giá cho %s từ %s đến %s...", ticker, start_date, end_date)

    raw = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        logger.warning("Không tải được dữ liệu giá cho %s trong khoảng thời gian đã cho.", ticker)
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # yfinance có thể trả về MultiIndex cột khi tải nhiều ticker cùng lúc,
    # dù ở đây chỉ tải 1 ticker - vẫn giữ safeguard này để tránh lỗi ngầm.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    logger.info("Tải xong %d dòng dữ liệu giá cho %s.", len(df), ticker)
    return df
