"""
Entry point để chạy pipeline. Minh họa khả năng mở rộng sang nhiều mã cổ
phiếu chỉ bằng cách thêm TickerConfig mới - không cần sửa logic pipeline.
"""

from config.settings import TickerConfig
from src.pipeline.build_dataset import build_dataset
from src.utils.logging_config import get_logger
from src.utils.validation import DataValidationError

logger = get_logger(__name__)


def main():
    configs = [
        TickerConfig(
            ticker="TSLA",
            company_name="Tesla",
            start_date="2023-01-01",
            end_date="2023-12-31",
        ),
        # Thêm mã cổ phiếu khác tại đây, ví dụ:
        # TickerConfig(
        #     ticker="AAPL",
        #     company_name="Apple",
        #     start_date="2023-01-01",
        #     end_date="2023-12-31",
        #     extra_aliases=["tim cook", "iphone maker"],
        # ),
    ]

    for cfg in configs:
        try:
            df = build_dataset(cfg)
            logger.info("Dataset %s: %s", cfg.ticker, df.shape)
        except DataValidationError as e:
            logger.error("Pipeline thất bại cho %s: %s", cfg.ticker, e)
            continue


if __name__ == "__main__":
    main()
