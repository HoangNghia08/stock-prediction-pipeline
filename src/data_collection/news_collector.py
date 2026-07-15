"""
Crawl tin tức tài chính qua GNews + tính relevance score.
"""

import html
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd
from gnews import GNews

from config.settings import (
    DEFAULT_NEWS_SOURCES,
    GNEWS_MAX_RESULTS,
    GNEWS_REQUEST_DELAY_SEC,
    HEADLINE_MATCH_WEIGHT,
    MAX_RELEVANCE_SCORE_CAP,
    MIN_RELEVANCE_SCORE,
    NEWS_CHUNK_SIZE_DAYS,
    SNIPPET_MATCH_WEIGHT,
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def generate_date_chunks(
    start_date: str, end_date: str, chunk_size_days: int = NEWS_CHUNK_SIZE_DAYS
) -> List[Tuple[str, str]]:
    """
    Chia khoảng [start_date, end_date] thành các đoạn nhỏ hơn (mặc định
    30 ngày/đoạn). 
    Vì Google News RSS giới hạn số kết quả cho mỗi truy vấn.
    Chia nhỏ giúp dữ liệu được đầy đủ hơn.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_size_days), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end

    return chunks


def _build_site_clause(sources: List[str]) -> str:
    return "(" + " OR ".join(f"site:{s}" for s in sources) + ")"


def clean_html_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_headline(raw_title: str, publisher_name: Optional[str]) -> str:
    """Cắt bỏ phần tên nguồn báo ở đuôi tiêu đề nếu có."""
    title = raw_title.strip()
    if publisher_name:
        suffix = f" - {publisher_name}"
        if title.endswith(suffix):
            return title[: -len(suffix)].strip()
    return title


def compute_relevance_score(headline: str, snippet: str, keywords: List[str]) -> int:
    """Tính điểm liên quan dựa trên số lần khớp từ khóa."""
    score = 0
    headline_lower = headline.lower()
    snippet_lower = snippet.lower()

    for kw in keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower:
            continue
        if kw_lower.startswith("$"):
            pattern = re.compile(re.escape(kw_lower))
        else:
            pattern = re.compile(r"\b" + re.escape(kw_lower) + r"\b")

        score += HEADLINE_MATCH_WEIGHT * len(pattern.findall(headline_lower))
        score += SNIPPET_MATCH_WEIGHT * len(pattern.findall(snippet_lower))

    return min(score, MAX_RELEVANCE_SCORE_CAP)


def collect_raw_news(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    keywords: List[str],
    source: Optional[str] = None,
    max_results: int = GNEWS_MAX_RESULTS,
    min_relevance_score: int = MIN_RELEVANCE_SCORE,
) -> pd.DataFrame:
    """
    Crawl tin tức thô cho 1 nguồn duy nhất trong 1 khoảng thời
    gian

    Trả về DataFrame với cột: Published_At_UTC, Headline, Snippet,
    Full_Text, Publisher, Relevance_Score, Source.
    """
    google_news = GNews(language="en", country="US")
    google_news.max_results = min(max_results, 100)

    if source:
        raw_query = f"{ticker} stock site:{source} after:{start_date} before:{end_date}"
    else:
        raw_query = f"{ticker} stock after:{start_date} before:{end_date}"

    try:
        news_results = google_news.get_news(raw_query)
    except Exception as e:
        logger.error(
            "Lỗi khi gọi Google News API [source=%s, %s -> %s]: %s",
            source, start_date, end_date, e,
        )
        return pd.DataFrame()

    # Log số kết quả THÔ (trước khi lọc relevance) - quan trọng để phân biệt
    # "GNews trả về 0 kết quả" (lỗi truy vấn/mạng) với "GNews trả về nhiều
    # kết quả nhưng đều bị lọc bỏ" (lỗi ngưỡng relevance quá chặt) khi debug.
    n_raw = len(news_results) if news_results else 0
    logger.debug("[source=%s, %s -> %s] GNews trả về %d kết quả thô.", source, start_date, end_date, n_raw)

    if not news_results:
        return pd.DataFrame()

    valid_records = []
    skipped_date_errors = 0
    skipped_irrelevant = 0
    skipped_out_of_range = 0

    chunk_start_dt = pd.Timestamp(start_date, tz="UTC")
    chunk_end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)

    for r in news_results:
        publisher_name = (
            r.get("publisher", {}).get("title")
            if isinstance(r.get("publisher"), dict)
            else None
        )
        headline = clean_headline(r.get("title", ""), publisher_name)
        snippet = clean_html_text(r.get("description", ""))

        if not headline:
            continue

        relevance_score = compute_relevance_score(headline, snippet, keywords)
        if relevance_score < min_relevance_score:
            skipped_irrelevant += 1
            continue

        try:
            pub_datetime = pd.to_datetime(r["published date"], utc=True)
        except (KeyError, ValueError, TypeError):
            skipped_date_errors += 1
            continue

        # Safeguard: after:/before: trên Google News đôi khi KHÔNG được
        # tuân thủ chặt chẽ (hạn chế đã biết) - tự lọc lại theo đúng
        # khoảng ngày yêu cầu để tránh lẫn dữ liệu sai đoạn thời gian.
        if not (chunk_start_dt <= pub_datetime < chunk_end_dt):
            skipped_out_of_range += 1
            continue

        valid_records.append(
            {
                "Published_At_UTC": pub_datetime,
                "Headline": headline,
                "Snippet": snippet,
                "Full_Text": f"{headline}. {snippet}".strip(),
                "Publisher": publisher_name,
                "Relevance_Score": relevance_score,
                "Source": source or "unrestricted",
            }
        )

    if skipped_irrelevant > 0 or skipped_date_errors > 0 or skipped_out_of_range > 0:
        logger.debug(
            "[source=%s, %s -> %s] Loại %d bài không liên quan, %d lỗi ngày, %d ngoài khoảng thời gian.",
            source, start_date, end_date, skipped_irrelevant, skipped_date_errors, skipped_out_of_range,
        )

    if not valid_records:
        return pd.DataFrame()

    return pd.DataFrame(valid_records)


def collect_raw_news_over_period(
    ticker: str,
    company_name: str,
    start_date: str,
    end_date: str,
    keywords: List[str],
    sources: List[str] = DEFAULT_NEWS_SOURCES,
    chunk_size_days: int = NEWS_CHUNK_SIZE_DAYS,
    request_delay_sec: float = GNEWS_REQUEST_DELAY_SEC,
    min_relevance_score: int = MIN_RELEVANCE_SCORE,
) -> pd.DataFrame:
    """
    Crawl tin tức cho TOÀN BỘ khoảng [start_date, end_date] bằng cách lặp
    LỒNG NHAU qua (từng đoạn thời gian) x (từng nguồn riêng biệt), mỗi lần
    gọi collect_raw_news() với truy vấn ĐƠN GIẢN.

    """
    date_chunks = generate_date_chunks(start_date, end_date, chunk_size_days)
    total_calls = len(date_chunks) * len(sources)
    logger.info(
        "Crawl tin tức cho %s: %d đoạn thời gian x %d nguồn = %d lượt gọi "
        "(ước tính tối thiểu %.0f giây do độ trễ giữa các lượt gọi)...",
        ticker, len(date_chunks), len(sources), total_calls, total_calls * request_delay_sec,
    )

    all_chunks_df = []
    call_count = 0

    for i, (chunk_start, chunk_end) in enumerate(date_chunks, start=1):
        for source in sources:
            call_count += 1
            logger.info(
                "  [%d/%d] Đoạn %s -> %s | Nguồn: %s",
                call_count, total_calls, chunk_start, chunk_end, source,
            )

            try:
                df_chunk = collect_raw_news(
                    ticker=ticker,
                    company_name=company_name,
                    start_date=chunk_start,
                    end_date=chunk_end,
                    keywords=keywords,
                    source=source,
                    min_relevance_score=min_relevance_score,
                )
            except Exception as e:
                logger.warning(
                    "    Lượt %s/%s thất bại, bỏ qua: %s", chunk_start, source, e
                )
                df_chunk = pd.DataFrame()

            if not df_chunk.empty:
                logger.info("    -> Lấy được %d bài.", len(df_chunk))
                all_chunks_df.append(df_chunk)

            time.sleep(request_delay_sec)  # tránh bị Google rate-limit

    if not all_chunks_df:
        logger.warning(
            "Không crawl được bài nào cho %s trong toàn bộ khoảng thời gian. "
            "Kiểm tra lại: (1) kết nối mạng, (2) thử chạy thủ công collect_raw_news() "
            "với 1 nguồn/1 đoạn thời gian nhỏ để xem log 'GNews trả về X kết quả thô' "
            "ở mức debug, giúp xác định GNews có trả kết quả hay không trước khi lọc.",
            ticker,
        )
        return pd.DataFrame()

    df_all = pd.concat(all_chunks_df, ignore_index=True)

    n_before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["Headline"]).reset_index(drop=True)
    n_after = len(df_all)
    if n_before != n_after:
        logger.info("Đã loại %d bài trùng lặp giữa các đoạn/nguồn.", n_before - n_after)

    logger.info("Crawl xong tổng cộng %d bản ghi tin tức thô cho %s.", n_after, ticker)
    if "Source" in df_all.columns:
        logger.info("Phân bố theo nguồn:\n%s", df_all["Source"].value_counts().to_string())

    return df_all