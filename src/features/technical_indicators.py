"""
Tính toán chỉ báo kỹ thuật (technical indicators) từ dữ liệu giá.

NGUYÊN TẮC THIẾT KẾ:
1. Tính nhân quả: mọi chỉ báo chỉ dùng dữ liệu tại thời điểm t và các thời
   điểm TRƯỚC t (rolling/ewm window nhìn về quá khứ) - không có rủi ro
   look-ahead bias, miễn là DataFrame đã sắp xếp tăng dần theo Date.
2. Ưu tiên tính DỪNG (stationarity): thay vì trả về SMA/Bollinger Band ở
   dạng giá trị TUYỆT ĐỐI (vẫn mang tính không dừng giống Close), hàm ưu
   tiên trả về dạng TƯƠNG ĐỐI (RSI, MACD Histogram, %B, tỷ lệ lệch so với
   SMA) - giúp mô hình tổng quát hóa tốt hơn giữa các giai đoạn giá khác
   nhau, giảm nguy cơ học theo kiểu persistence (ŷ ≈ y_{t-1}).
3. Có "giai đoạn khởi động" (warm-up period): các chỉ báo cần N ngày dữ
   liệu quá khứ để tính (ví dụ RSI-14 cần 14 ngày) sẽ cho ra NaN ở N dòng
   đầu tiên - đây là điều BÌNH THƯỜNG, không phải lỗi, xử lý bằng dropna()
   ở bước cuối (đã có sẵn log số dòng bị xóa qua validate_merged_dataframe).
"""

import numpy as np
import pandas as pd

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def add_log_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Log-return: log(Close_t / Close_{t-1}) - có tính dừng tốt hơn nhiều so
    với giá tuyệt đối, nên cân nhắc dùng thay/kèm Close khi đưa vào model.
    """
    df = df.copy()
    df["Log_Return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def add_sma_ratio(df: pd.DataFrame, price_col: str = "Close", windows=(7, 30)) -> pd.DataFrame:
    """
    Tỷ lệ lệch so với SMA: Close/SMA_n - 1, thay vì trả SMA thô (vốn không
    dừng, dễ gây persistence learning). Giá trị dương = giá đang cao hơn
    trung bình n ngày gần nhất (xu hướng tăng); âm = đang thấp hơn.
    """
    df = df.copy()
    for n in windows:
        sma = df[price_col].rolling(window=n, min_periods=n).mean()
        df[f"SMA_{n}_Ratio"] = df[price_col] / sma - 1
    return df


def add_rsi(df: pd.DataFrame, price_col: str = "Close", window: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) - luôn nằm trong [0, 100], có tính dừng
    tốt tự nhiên. RSI > 70 thường coi là quá mua (overbought), < 30 quá
    bán (oversold).

    Công thức chuẩn (Wilder, 1978):
        RS  = trung bình tăng giá n ngày / trung bình giảm giá n ngày
        RSI = 100 - 100 / (1 + RS)
    """
    df = df.copy()
    delta = df[price_col].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)  # tránh chia cho 0
    rsi = 100 - (100 / (1 + rs))

    # Nếu avg_loss = 0 (giá chỉ tăng liên tục trong window) -> RSI = 100
    rsi = rsi.fillna(100).where(avg_loss != 0, 100)
    df[f"RSI_{window}"] = rsi
    return df


def add_macd(
    df: pd.DataFrame, price_col: str = "Close", fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    MACD Histogram - dao động quanh 0, có tính dừng tốt hơn MACD tuyệt đối.

    Công thức:
        MACD_line = EMA_fast(Close) - EMA_slow(Close)
        Signal_line = EMA_signal(MACD_line)
        MACD_Histogram = MACD_line - Signal_line

    Giá trị dương = động lượng tăng đang mạnh lên; âm = động lượng giảm.
    """
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    df["MACD_Histogram"] = macd_line - signal_line
    return df


def add_bollinger_percent_b(
    df: pd.DataFrame, price_col: str = "Close", window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    %B của Bollinger Band - vị trí tương đối của giá trong dải Bollinger,
    thường nằm trong khoảng [0, 1] (có thể vượt ra ngoài khi giá phá dải).

    Công thức:
        Middle = SMA_n(Close)
        Upper  = Middle + k * std_n(Close)
        Lower  = Middle - k * std_n(Close)
        %B     = (Close - Lower) / (Upper - Lower)

    %B gần 1 = giá gần dải trên (có thể quá mua); gần 0 = giá gần dải dưới.
    """
    df = df.copy()
    middle = df[price_col].rolling(window=window, min_periods=window).mean()
    std = df[price_col].rolling(window=window, min_periods=window).std()

    upper = middle + num_std * std
    lower = middle - num_std * std

    df["Bollinger_PercentB"] = (df[price_col] - lower) / (upper - lower)
    return df


def add_rolling_volatility(df: pd.DataFrame, price_col: str = "Close", windows=(7, 14)) -> pd.DataFrame:
    """
    Độ biến động (volatility): độ lệch chuẩn của log-return trong cửa sổ n
    ngày - đo mức độ "bất ổn" gần đây của giá, không phụ thuộc vào mức giá
    tuyệt đối (có tính dừng tốt).
    """
    df = df.copy()
    if "Log_Return" not in df.columns:
        df = add_log_return(df, price_col=price_col)

    for n in windows:
        df[f"Volatility_{n}"] = df["Log_Return"].rolling(window=n, min_periods=n).std()

    return df


def add_all_technical_indicators(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    Áp dụng toàn bộ các chỉ báo kỹ thuật được khuyến nghị (bộ tối giản,
    tránh đa cộng tuyến - mỗi loại thông tin chỉ giữ 1 đại diện chính):

    - Log_Return: xu hướng ngắn hạn dạng có tính dừng
    - SMA_7_Ratio, SMA_30_Ratio: xu hướng ngắn/dài hạn dạng tương đối
    - RSI_14: động lượng
    - MACD_Histogram: động lượng/xu hướng
    - Bollinger_PercentB: vị trí giá trong biên độ dao động gần đây
    - Volatility_7: mức độ biến động ngắn hạn

    DataFrame đầu vào PHẢI đã sắp xếp tăng dần theo Date (bắt buộc để các
    phép rolling/ewm tính đúng theo đúng thứ tự thời gian, đảm bảo tính
    nhân quả - không có look-ahead bias).
    """
    if "Date" in df.columns:
        dates = pd.to_datetime(df["Date"])
        if not dates.is_monotonic_increasing:
            logger.warning("DataFrame chưa sắp xếp tăng dần theo Date - sắp xếp lại trước khi tính chỉ báo.")
            df = df.sort_values("Date").reset_index(drop=True)

    n_before = len(df)

    df = add_log_return(df, price_col)
    df = add_sma_ratio(df, price_col, windows=(7, 30))
    df = add_rsi(df, price_col, window=14)
    df = add_macd(df, price_col)
    df = add_bollinger_percent_b(df, price_col)
    df = add_rolling_volatility(df, price_col, windows=(7,))

    new_cols = [
        "Log_Return", "SMA_7_Ratio", "SMA_30_Ratio",
        "RSI_14", "MACD_Histogram", "Bollinger_PercentB", "Volatility_7",
    ]
    n_nan_rows = df[new_cols].isna().any(axis=1).sum()
    logger.info(
        "Đã thêm %d chỉ báo kỹ thuật. %d/%d dòng đầu tiên có NaN (giai đoạn "
        "khởi động, cần loại bỏ bằng dropna() trước khi đưa vào model).",
        len(new_cols), n_nan_rows, n_before,
    )

    return df
