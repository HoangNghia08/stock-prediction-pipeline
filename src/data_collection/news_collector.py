"""
Crawl tin tức tài chính qua GNews + tính relevance score (mức độ liên quan của tin tức với mã cổ phiếu đang xét)
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
    - Chia khoảng [start_date, end_date] thành các đoạn nhỏ (mặc định 30 ngày/đoạn).
    - Lý do: Google News RSS giới hạn số kết quả cho mỗi truy vấn (100 kết quả cho 1 lần truy vấn). Nên nếu thực hiện duy nhất
    1 truy vấn cho toàn bộ thời gian, dữ liệu sẽ không thể đầy đủ.
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
    """Tính điểm liên quan dựa trên số lần khớp từ khóa (có ranh giới từ)."""
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
    sources: List[str] = DEFAULT_NEWS_SOURCES,
    max_results: int = GNEWS_MAX_RESULTS,
    min_relevance_score: int = MIN_RELEVANCE_SCORE,
) -> pd.DataFrame:
    """
    Crawl tin tức thô từ nhiều nguồn cùng lúc trong 1 khoảng thời gian, + tính relevance_score.
    Trả về DataFrame với cột: Published_At_UTC, Headline, Snippet,
    Full_Text, Publisher, Relevance_Score.
    """
    google_news = GNews(language="en", country="US")
    google_news.max_results = min(max_results, 100)

    site_clause = _build_site_clause(sources)
    raw_query = (
        f"({ticker} OR {company_name}) {site_clause} "
        f"after:{start_date} before:{end_date}"
    )

    try:
        news_results = google_news.get_news(raw_query)
    except Exception as e:
        logger.error("Lỗi khi gọi Google News API cho [%s -> %s]: %s", start_date, end_date, e)
        return pd.DataFrame()

    if not news_results:
        return pd.DataFrame()

    valid_records = []
    skipped_date_errors = 0
    skipped_irrelevant = 0

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

        valid_records.append(
            {
                "Published_At_UTC": pub_datetime,
                "Headline": headline,
                "Snippet": snippet,
                "Full_Text": f"{headline}. {snippet}".strip(),
                "Publisher": publisher_name,
                "Relevance_Score": relevance_score,
            }
        )

    if skipped_irrelevant > 0 or skipped_date_errors > 0:
        logger.debug(
            "[%s -> %s] Loại %d bài không liên quan, %d bài lỗi ngày.",
            start_date, end_date, skipped_irrelevant, skipped_date_errors,
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
    Crawl tin tức cho TOÀN BỘ khoảng [start_date, end_date] bằng cách chia
    thành nhiều đoạn nhỏ (mặc định 30 ngày/đoạn) và gọi collect_raw_news()
    riêng cho từng đoạn, rồi gộp + khử trùng lặp.
    """
    date_chunks = generate_date_chunks(start_date, end_date, chunk_size_days)
    logger.info(
        "Crawl tin tức cho %s từ %s nguồn, chia thành %d đoạn thời gian (~%d ngày/đoạn)...",
        ticker, len(sources), len(date_chunks), chunk_size_days,
    )

    all_chunks_df = []
    for i, (chunk_start, chunk_end) in enumerate(date_chunks, start=1):
        logger.info("  Đoạn %d/%d: %s -> %s", i, len(date_chunks), chunk_start, chunk_end)

        try:
            df_chunk = collect_raw_news(
                ticker=ticker,
                company_name=company_name,
                start_date=chunk_start,
                end_date=chunk_end,
                keywords=keywords,
                sources=sources,
                min_relevance_score=min_relevance_score,
            )
        except Exception as e:
            logger.warning("  Đoạn %s -> %s thất bại, bỏ qua: %s", chunk_start, chunk_end, e)
            df_chunk = pd.DataFrame()

        if not df_chunk.empty:
            logger.info("    -> Lấy được %d bài.", len(df_chunk))
            all_chunks_df.append(df_chunk)

        time.sleep(request_delay_sec)  # tránh bị Google rate-limit khi gọi liên tục nhiều đoạn

    if not all_chunks_df:
        logger.warning("Không crawl được bài nào cho %s trong toàn bộ khoảng thời gian.", ticker)
        return pd.DataFrame()

    df_all = pd.concat(all_chunks_df, ignore_index=True)

    # Khử trùng lặp TOÀN CỤC - vì các đoạn thời gian liền kề có thể vô tình
    # trả về cùng 1 bài (ví dụ bài đăng sát ranh giới giữa 2 đoạn).
    n_before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["Headline"]).reset_index(drop=True)
    n_after = len(df_all)
    if n_before != n_after:
        logger.info("Đã loại %d bài trùng lặp giữa các đoạn thời gian.", n_before - n_after)

    logger.info("Crawl xong tổng cộng %d bản ghi tin tức thô cho %s.", n_after, ticker)
    return df_all
