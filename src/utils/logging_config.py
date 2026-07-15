"""Logging thống nhất cho toàn bộ project - không dùng print() ở bất kỳ đâu khác."""

import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name)
