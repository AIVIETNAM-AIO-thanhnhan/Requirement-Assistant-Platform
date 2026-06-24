# Chunk Tuning - Grid Search

Mục tiêu: tìm CHUNK_SIZE / CHUNK_OVERLAP tối ưu, đo trên gold 38 câu, so mốc BASELINE.md.
Công cụ: scripts/tune_chunking.py (mỗi cấu hình build collection tạm, cùng embedder
AITeamVN/Vietnamese_Embedding + metric L2 như baseline).

Ngày chạy: 2026-06-24

## Sweep 1 — size (overlap cố định 150)

| size | overlap | chunks | R@1    | R@3    | R@5    | MRR    |
| ---- | ------- | ------ | ------ | ------ | ------ | ------ |
| 600  | 150     | 1173   | 0.7105 | 0.8947 | 0.8947 | 0.8026 |
| 800  | 150     | 839    | 0.6579 | 0.8421 | 0.9474 | 0.7618 |
| 1000 | 150     | 646    | 0.8158 | 0.9474 | 0.9737 | 0.8781 |
| 1200 | 150     | 519    | 0.6316 | 0.8947 | 0.9211 | 0.7610 |

## Sweep 2 — overlap (size cố định 1000)

| size | overlap | chunks | R@1    | R@3    | R@5    | MRR    |
| ---- | ------- | ------ | ------ | ------ | ------ | ------ |
| 1000 | 50      | 585    | 0.7632 | 0.8947 | 0.9211 | 0.8224 |
| 1000 | 100     | 617    | 0.6316 | 0.9474 | 0.9474 | 0.7719 |
| 1000 | 150     | 646    | 0.8158 | 0.9474 | 0.9737 | 0.8781 |
| 1000 | 200     | 672    | 0.7895 | 0.8947 | 0.9737 | 0.8575 |
| 1000 | 250     | 699    | 0.8158 | 0.9211 | 0.9474 | 0.8737 |

## Kết luận

- **Giữ nguyên CHUNK_SIZE=1000, CHUNK_OVERLAP=150.** Đây là cấu hình tốt nhất:
  cao nhất MRR (0.8781), đồng hạng nhất R@1 (0.8158) và R@5 (0.9737). Không cấu
  hình nào trong lưới vượt qua.
- Cấu hình 1000/150 tái lập CHÍNH XÁC baseline → xác nhận tuner tái dựng pipeline đúng.
- size nhỏ (600/800) làm vỡ ranh giới Điều → giảm chính xác; size lớn (1200) gộp nhiều
  Điều → nhiễu. overlap quanh 150 là điểm cân bằng.

## Cảnh báo độ tin cậy

Gold 38 câu: lệch 1 câu ≈ 2.6 điểm phần trăm. Các dao động nhỏ giữa cấu hình (vd
overlap=100 tụt bất thường) nằm trong khoảng nhiễu, KHÔNG nên diễn giải quá mức. Kết
luận chắc chắn: 1000/150 ổn định ở/gần đỉnh trên mọi metric.

## Hướng cải thiện thực sự (không phải chunk size)

Điểm yếu còn lại (vd q14 Điều 109 MISS) đến từ việc cùng chủ đề nằm ở nhiều văn bản
(Bộ luật nêu ngắn, Nghị định hướng dẫn chi tiết). Hướng tiếp theo: enrich metadata /
lọc-ưu tiên theo document_type, không phải tinh chỉnh kích thước chunk.
