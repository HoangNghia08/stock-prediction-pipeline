# Stock Prediction Data Pipeline

Pipeline thu thập dữ liệu giá cổ phiếu + sentiment tin tức (FinBERT) làm input
cho mô hình dự báo giá (CNN-BiLSTM-Attention).

## Cách chạy

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Cấu trúc

- `config/settings.py` - toàn bộ cấu hình (ticker aliases, model, đường dẫn)
- `src/data_collection/` - thu thập giá (yfinance) và tin tức thô (GNews)
- `src/sentiment/` - phân tích cảm xúc bằng FinBERT
- `src/processing/` - tổng hợp theo ngày + căn chỉnh lịch giao dịch
- `src/pipeline/` - orchestration, ghép nối toàn bộ các bước trên
- `data/raw/` - checkpoint dữ liệu thô (giá, tin tức) - KHÔNG bị ghi đè
- `data/interim/` - dữ liệu đã chấm sentiment nhưng chưa merge
- `data/processed/` - dataset cuối cùng (.parquet), sẵn sàng cho bước tạo sliding window

## Lưu ý quan trọng

- `auto_adjust=True` khi tải giá - BẮT BUỘC để tránh bước nhảy giá giả do stock split.
- Tin tức đăng sau giờ đóng cửa (>=16:00 ET) hoặc cuối tuần được dồn sang
  ngày giao dịch kế tiếp để tránh look-ahead bias.
- `Has_News` là cờ nhị phân phân biệt "không có tin" với "có tin nhưng trung tính".
