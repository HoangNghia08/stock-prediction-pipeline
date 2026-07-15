"""
Wrapper cho FinBERT - tách biệt hoàn toàn khỏi bước crawl tin tức, để có
thể thay thế mô hình sentiment
"""

from typing import List

import pandas as pd

from config.settings import FINBERT_BATCH_SIZE, FINBERT_MODEL_NAME
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FinBERTSentimentAnalyzer:
    """Lazy-load FinBERT 1 lần duy nhất, tái sử dụng cho nhiều lần gọi."""

    def __init__(self, model_name: str = FINBERT_MODEL_NAME):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline

            try:
                import torch
                device = 0 if torch.cuda.is_available() else -1
            except ImportError:
                device = -1

            logger.info(
                "Đang tải mô hình %s (device=%s)...",
                self.model_name, "GPU" if device == 0 else "CPU",
            )
            self._pipeline = hf_pipeline("sentiment-analysis", model=self.model_name, device=device)
            logger.info("Tải mô hình sentiment hoàn tất!")
        return self._pipeline

    def score_texts(self, texts: List[str], batch_size: int = FINBERT_BATCH_SIZE) -> pd.DataFrame:
        """
        Chấm điểm sentiment cho danh sách văn bản.

        Trả về DataFrame với 3 cột: Sentiment_Label, Confidence_Score,
        Signed_Sentiment_Score (dương nếu Positive, âm nếu Negative, 0 nếu
        Neutral - giữ lại cả hướng lẫn cường độ cảm xúc).
        """
        if not texts:
            return pd.DataFrame(columns=["Sentiment_Label", "Confidence_Score", "Signed_Sentiment_Score"])

        model = self._load()
        results = model(texts, truncation=True, batch_size=batch_size)

        labels = [res["label"].capitalize() for res in results]
        confidences = [res["score"] for res in results]
        signed_scores = [
            conf if label == "Positive" else (-conf if label == "Negative" else 0.0)
            for label, conf in zip(labels, confidences)
        ]

        return pd.DataFrame(
            {
                "Sentiment_Label": labels,
                "Confidence_Score": confidences,
                "Signed_Sentiment_Score": signed_scores,
            }
        )
