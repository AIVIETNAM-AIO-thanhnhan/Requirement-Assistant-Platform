# Baseline Retrieval Evaluation

Mốc tham chiếu cho mọi thử nghiệm chunk-tuning về sau. KHÔNG sửa file này khi tune;
tạo bản ghi mới (vd BASELINE_v2.md) và so với mốc này.

Ngày đo: 2026-06-21

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
|                     | BoLuatLaoDong_2019: 294 (luat)                       |
|                     | NghiDinh_145_2020: 289 (nghi_dinh)                   |
|                     | ThongTu_10_2020_BLDTBXH: 45 (thong_tu)              |
|                     | NghiDinh_135_2020: 18 (nghi_dinh)                    |
| Gold set            | eval/gold_match.jsonl (match-based, 6 câu)           |
| top_k khi đo        | 5                                                    |

## Kết quả

| Metric    | Giá trị |
| --------- | ------- |
| Recall@1  | 0.5000  |
| Recall@3  | 0.8333  |
| Recall@5  | 1.0000  |
| MRR       | 0.6528  |

Chi tiết theo câu (hạng chunk đúng đầu tiên):

| ID | Hạng | RR     | Câu hỏi                                          |
| -- | ---- | ------ | ------------------------------------------------ |
| q1 | 4    | 0.2500 | Thời giờ làm việc bình thường tối đa mỗi tuần    |
| q2 | 3    | 0.3333 | Mỗi ngày làm việc bình thường tối đa mấy giờ     |
| q3 | 1    | 1.0000 | Số giờ làm thêm tối đa trong một năm             |
| q4 | 3    | 0.3333 | Nghỉ giữa giờ ít nhất bao nhiêu phút             |
| q5 | 1    | 1.0000 | Nghỉ việc riêng khi kết hôn mấy ngày             |
| q6 | 1    | 1.0000 | Các ngày nghỉ lễ tết hưởng nguyên lương          |

## Cảnh báo về độ tin cậy

Gold set hiện chỉ có **6 câu** → số liệu mang tính chỉ báo, chưa đủ vững để ra quyết
định tuning. Cần mở rộng lên **≥30–50 câu** trải đều 4 văn bản trước khi coi đây là
mốc chính thức để so sánh.
