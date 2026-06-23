# Baseline Retrieval Evaluation (chính thức)

Mốc tham chiếu cho mọi thử nghiệm chunk-tuning về sau. KHÔNG sửa file này khi tune;
tạo bản ghi mới (vd BASELINE_v2.md) và so với mốc này.

Ngày đo: 2026-06-24
Ghi chú: thay thế baseline tạm 6 câu trước đó (R@1=0.50) bằng bộ gold 38 câu cân đối.

## Cấu hình đã dùng (phải tái lập đúng để so sánh)

| Hạng mục            | Giá trị                                              |
| ------------------- | ---------------------------------------------------- |
| Embedder            | AITeamVN/Vietnamese_Embedding (nền BGE-M3)           |
| Số chiều embedding  | 1024 (dot-product similarity)                        |
| EMBED_PROVIDER      | bge                                                  |
| Chunk size          | 1000 ký tự                                           |
| Chunk overlap       | 150 ký tự (hardcode trong process_documents.py)      |
| Separator ưu tiên   | \nĐiều , \nChương , \n\n, \n, ". ", " "              |
| Chuẩn hóa text      | NFC (clean_text)                                     |
| Collection          | qa_documents_vi                                      |
| Corpus              | 4 văn bản → 646 chunk                                |
| Gold set            | eval/gold_match.jsonl (match-based, 38 câu)          |
| Phân bố gold        | 19 Bộ luật / 11 NĐ145 / 3 NĐ135 / 5 TT10            |
| top_k khi đo        | 5                                                    |

## Kết quả

| Metric    | Giá trị |
| --------- | ------- |
| Recall@1  | 0.8158  |
| Recall@3  | 0.9474  |
| Recall@5  | 0.9737  |
| MRR       | 0.8781  |

31/38 câu đạt hạng 1. Các câu chưa tối ưu (hạng > 1):

| ID  | Điều / nguồn            | Hạng | Ghi chú                                   |
| --- | ----------------------- | ---- | ----------------------------------------- |
| q5  | Điều 36 (Bộ luật)       | 3    |                                           |
| q8  | Điều 54 (Bộ luật)       | 3    | cạnh tranh với Điều 53/55 cho thuê lại    |
| q14 | Điều 109 (Bộ luật)      | MISS | bị NĐ145 Điều 58/64 (cùng chủ đề) lấn át  |
| q21 | Điều 8 (NĐ145)          | 5    |                                           |
| q29 | Điều 69 (NĐ145)         | 2    |                                           |
| q33 | Điều 7 (NĐ135)          | 2    |                                           |
| q34 | Điều 3 (TT10)           | 2    |                                           |

## Quan sát chính

- Điểm yếu rõ nhất: câu hỏi mà nội dung trùng chủ đề ở nhiều văn bản (Bộ luật nêu
  ngắn, Nghị định hướng dẫn chi tiết) → retrieval ưu tiên bản chi tiết, lấn át điều
  luật gốc (điển hình q14). Đây là mục tiêu cho enrich metadata / điều chỉnh chunk
  ở Phase 2, KHÔNG phải lỗi gold.
- Toàn bộ 38 câu đã qua validate_gold (mọi marker đều có chunk khớp trong corpus).
