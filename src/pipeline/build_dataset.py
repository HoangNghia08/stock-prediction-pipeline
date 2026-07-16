"""
Orchestration chính của pipeline: điều phối price_collector, news_collector,
finbert_analyzer, calendar_alignment, decay -> trả về dataset cuối cùng.
"""

from datetime import datetime

import pandas as pd

from config.settings import DATA_INTERIM_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, TickerConfig
from src.data_collection.news_collector import collect_raw_news_over_period
from src.data_collection.price_collector import collect_price_data
from src.features.decay import apply_sentiment_decay
from src.processing.calendar_alignment import aggregate_daily_sentiment
from src.sentiment.finbert_analyzer import FinBERTSentimentAnalyzer
from src.utils.logging_config import get_logger
from src.utils.validation import DataValidationError, validate_merged_dataframe, validate_price_dataframe

logger = get_logger(__name__)

SENTIMENT_FILL_ZERO_COLS = [
    "News_Count", "Positive_Count", "Negative_Count", "Neutral_Count",
    "Avg_Relevance_Score", "Weighted_Sentiment_Score", "Net_Sentiment_Score", "Has_News",
]


def build_dataset(config: TickerConfig, save_checkpoints: bool = True) -> pd.DataFrame:
    """
    Xây dựng dataset cuối cùng (giá + sentiment theo ngày) cho 1 mã cổ phiếu.
    """
    run_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # ---- 1. Dữ liệu giá ----
    df_price = collect_price_data(config.ticker, config.start_date, config.end_date)
    validate_price_dataframe(df_price)

    if save_checkpoints:
        raw_price_path = DATA_RAW_DIR / f"price_{config.ticker}_{run_timestamp}.csv"
        df_price.to_csv(raw_price_path, index=False)
        logger.info("Đã lưu checkpoint giá thô: %s", raw_price_path)

    # ---- 2. Dữ liệu tin tức thô (đa nguồn, chia nhỏ theo tháng) ----
    df_news_raw = collect_raw_news_over_period(
        ticker=config.ticker,
        company_name=config.company_name,
        start_date=config.start_date,
        end_date=config.end_date,
        keywords=config.all_aliases(),
    )

    if df_news_raw.empty:
        logger.warning(
            "KHÔNG crawl được tin tức nào cho %s. Dataset sẽ có toàn bộ "
            "feature sentiment = 0/Has_News = 0.",
            config.ticker,
        )
        df_sentiment_daily = pd.DataFrame(columns=["Date"] + SENTIMENT_FILL_ZERO_COLS)
    else:
        if save_checkpoints:
            raw_news_path = DATA_RAW_DIR / f"news_{config.ticker}_{run_timestamp}.csv"
            df_news_raw.to_csv(raw_news_path, index=False)
            logger.info("Đã lưu checkpoint tin tức thô: %s", raw_news_path)

        # ---- 3. Sentiment ----
        analyzer = FinBERTSentimentAnalyzer()
        try:
            sentiment_df = analyzer.score_texts(df_news_raw["Full_Text"].tolist())
            df_news_scored = pd.concat(
                [df_news_raw.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1
            )
        except Exception as e:
            logger.error("FinBERT thất bại: %s. Dữ liệu thô đã được lưu.", e)
            df_sentiment_daily = pd.DataFrame(columns=["Date"] + SENTIMENT_FILL_ZERO_COLS)
        else:
            if save_checkpoints:
                scored_path = DATA_INTERIM_DIR / f"news_scored_{config.ticker}_{run_timestamp}.csv"
                df_news_scored.to_csv(scored_path, index=False)
                logger.info("Đã lưu checkpoint tin tức đã chấm sentiment: %s", scored_path)

            # ---- 4. Tổng hợp theo ngày + căn chỉnh lịch giao dịch ----
            df_sentiment_daily = aggregate_daily_sentiment(df_news_scored)

    # ---- 5. Merge giá + sentiment ----
    df_merged = pd.merge(df_price, df_sentiment_daily, on="Date", how="left")
    df_merged[SENTIMENT_FILL_ZERO_COLS] = df_merged[SENTIMENT_FILL_ZERO_COLS].fillna(0)

    # ---- 6. Sentiment Decay: mô phỏng hiệu ứng tin tức suy giảm dần theo
    # thời gian, thay vì ảnh hưởng biến mất hoàn toàn ngay ngày hôm sau.
    # BẮT BUỘC thực hiện SAU merge+fillna (không phải trên bảng tin tức thô)
    # để không bị "nhảy cóc" qua các ngày không có tin - xem chi tiết trong
    # docstring của src/features/decay.py.
    df_merged = apply_sentiment_decay(df_merged)

    n_before_dropna = len(df_merged)
    df_merged = df_merged.dropna().reset_index(drop=True)
    validate_merged_dataframe(df_merged, n_before_dropna)

    if save_checkpoints:
        output_path = DATA_PROCESSED_DIR / f"dataset_{config.ticker}_{config.start_date}_{config.end_date}.parquet"
        df_merged.to_parquet(output_path, index=False)
        logger.info("Đã lưu dataset cuối cùng: %s", output_path)

    logger.info("Hoàn tất pipeline cho %s: %d dòng, %d cột.", config.ticker, *df_merged.shape)
    return df_merged

def _generate_year_ranges(start_date: str, end_date: str) -> list:
    """Chia [start_date, end_date] thành các đoạn theo TỪNG NĂM DƯƠNG LỊCH."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    ranges = []
    for year in range(start.year, end.year + 1):
        year_start = max(datetime(year, 1, 1), start)
        year_end = min(datetime(year, 12, 31), end)
        if year_start <= year_end:
            ranges.append((year, year_start.strftime("%Y-%m-%d"), year_end.strftime("%Y-%m-%d")))
    return ranges


def _crawl_and_score_one_year(
    ticker: str, company_name: str, keywords: list,
    year: int, sub_start: str, sub_end: str,
) -> pd.DataFrame:
    """Crawl + chấm sentiment cho ĐÚNG 1 năm - dùng nội bộ bởi build_dataset_long_range."""
    df_news_raw = collect_raw_news_over_period(
        ticker=ticker, company_name=company_name,
        start_date=sub_start, end_date=sub_end, keywords=keywords,
    )

    if df_news_raw.empty:
        logger.warning("Năm %d: không crawl được tin tức nào.", year)
        return pd.DataFrame()

    analyzer = FinBERTSentimentAnalyzer()
    try:
        sentiment_df = analyzer.score_texts(df_news_raw["Full_Text"].tolist())
    except Exception as e:
        logger.error("Năm %d: FinBERT thất bại (%s) - bỏ qua năm này, dữ liệu thô KHÔNG mất "
                     "(chưa lưu checkpoint nên cần crawl lại lần sau).", year, e)
        return pd.DataFrame()

    return pd.concat(
        [df_news_raw.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1
    )


def build_dataset_long_range(config: TickerConfig, save_checkpoints: bool = True) -> pd.DataFrame:
    """
    Xây dựng dataset cho khoảng thời gian DÀI (nhiều năm), chia crawl theo
    TỪNG NĂM và lưu checkpoint riêng cho mỗi năm - có khả năng RESUME.
    """
    keywords = config.all_aliases()
    year_ranges = _generate_year_ranges(config.start_date, config.end_date)
    logger.info("Sẽ crawl tin tức cho %d năm: %s", len(year_ranges), [y for y, _, _ in year_ranges])

    all_years_news = []
    for year, sub_start, sub_end in year_ranges:
        checkpoint_path = DATA_INTERIM_DIR / f"news_scored_{config.ticker}_{year}.csv"

        if checkpoint_path.exists():
            logger.info("Năm %d: đã có checkpoint (%s) - đọc lại từ đĩa, BỎ QUA crawl.", year, checkpoint_path.name)
            df_year = pd.read_csv(checkpoint_path)
            if "Published_At_UTC" in df_year.columns:
                df_year["Published_At_UTC"] = pd.to_datetime(df_year["Published_At_UTC"], utc=True)
        else:
            logger.info("Năm %d: chưa có checkpoint - bắt đầu crawl (%s -> %s)...", year, sub_start, sub_end)
            df_year = _crawl_and_score_one_year(config.ticker, config.company_name, keywords, year, sub_start, sub_end)
            if save_checkpoints:
                df_year.to_csv(checkpoint_path, index=False)
                logger.info("Năm %d: đã lưu checkpoint (%d bài) -> %s", year, len(df_year), checkpoint_path.name)

        if not df_year.empty:
            all_years_news.append(df_year)

    df_news_scored = pd.concat(all_years_news, ignore_index=True) if all_years_news else pd.DataFrame()
    logger.info("Tổng cộng %d bài tin tức đã chấm sentiment trên toàn bộ %d năm.", len(df_news_scored), len(year_ranges))

    df_price = collect_price_data(config.ticker, config.start_date, config.end_date)
    validate_price_dataframe(df_price)

    if save_checkpoints:
        raw_price_path = DATA_RAW_DIR / f"price_{config.ticker}_{config.start_date}_{config.end_date}.csv"
        df_price.to_csv(raw_price_path, index=False)

    if df_news_scored.empty:
        df_sentiment_daily = pd.DataFrame(columns=["Date"] + SENTIMENT_FILL_ZERO_COLS)
    else:
        df_sentiment_daily = aggregate_daily_sentiment(df_news_scored)

    df_merged = pd.merge(df_price, df_sentiment_daily, on="Date", how="left")
    df_merged[SENTIMENT_FILL_ZERO_COLS] = df_merged[SENTIMENT_FILL_ZERO_COLS].fillna(0)
    df_merged = apply_sentiment_decay(df_merged)

    n_before_dropna = len(df_merged)
    df_merged = df_merged.dropna().reset_index(drop=True)
    validate_merged_dataframe(df_merged, n_before_dropna)

    if save_checkpoints:
        output_path = DATA_PROCESSED_DIR / f"dataset_{config.ticker}_{config.start_date}_{config.end_date}.parquet"
        df_merged.to_parquet(output_path, index=False)
        logger.info("Đã lưu dataset cuối cùng (nhiều năm): %s", output_path)

    logger.info("Hoàn tất pipeline dài hạn cho %s: %d dòng, %d cột.", config.ticker, *df_merged.shape)
    return df_merged
