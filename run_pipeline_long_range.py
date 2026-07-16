"""
Script chạy pipeline cho khoảng thời gian DÀI (2019-2026), thiết kế để chạy
trên Google Colab với khả năng RESUME nếu bị ngắt kết nối giữa chừng.
"""

from config.settings import TickerConfig
from src.pipeline.build_dataset import build_dataset_long_range
from src.utils.logging_config import get_logger
from src.utils.validation import DataValidationError

logger = get_logger(__name__)


def main():
    config = TickerConfig(
        ticker="TSLA",
        company_name="Tesla",
        start_date="2019-01-01",
        end_date="2026-07-15",
    )

    try:
        df = build_dataset_long_range(config)
        logger.info("HOÀN TẤT. Dataset cuối cùng: %s", df.shape)
        print(df.head(10))
        print(df.tail(10))
    except DataValidationError as e:
        logger.error("Pipeline thất bại: %s", e)


if __name__ == "__main__":
    main()
