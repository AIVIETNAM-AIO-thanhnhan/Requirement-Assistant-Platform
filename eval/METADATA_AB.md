# Metadata Re-rank A/B — document_type

Câu hỏi: lọc/ưu tiên theo document_type khi truy vấn có cải thiện retrieval không?
Công cụ: scripts/metadata_rerank_ab.py (dùng lại collection baseline qa_documents_vi,
lấy top-20 ứng viên rồi rerank; không build lại DB). Gold 38 câu.

Ngày chạy: 2026-06-24

## Kết quả

| scheme      | R@1    | R@3    | R@5    | MRR    |
| ----------- | ------ | ------ | ------ | ------ |
| baseline    | 0.8158 | 0.9474 | 0.9737 | 0.8807 |
| boost_luat  | 0.4211 | 0.5000 | 0.5789 | 0.5177 |
| oracle      | 0.8684 | 0.9737 | 1.0000 | 0.9219 |

- baseline : thứ tự theo khoảng cách (như rag_eval).
- boost_luat: đẩy mọi chunk document_type='luat' lên trước (heuristic triển khai được).
- oracle   : đẩy đúng loại văn bản của đáp án lên trước (TRẦN lý thuyết, không
             triển khai được vì cần biết trước nguồn đúng).

## Kết luận: KHÔNG áp dụng re-rank theo document_type

1. **boost_luat có hại nặng** (R@1 0.8158 → 0.4211). Trên corpus trộn nhiều nguồn
   (gold: 19 luat / 11 nghi_dinh / 3 nghi_dinh / 5 thong_tu), ưu tiên một loại văn
   bản cứng nhắc chôn vùi đáp án đúng của các câu thuộc Nghị định/Thông tư.
2. **oracle chỉ hơn khiêm tốn**: R@1 +5.3 điểm = đúng 2 câu trên 38 (nằm trong vùng
   nhiễu ±2.6đ/câu). Lại không triển khai được nếu không có bộ phân loại câu hỏi →
   loại văn bản — chi phí không tương xứng với mức lợi.

## Hệ quả

- Giữ retrieval hiện tại (size=1000, overlap=150, không rerank theo doctype).
- Điểm yếu còn lại (q14: Điều 109 vs NĐ145 cùng chủ đề) phần nào là ranh giới gán
  nhãn gold — đáp án Nghị định cũng hợp lý. Không phải lỗi cần "sửa" bằng metadata.
- Nếu sau này muốn chạm trần oracle: hướng đúng là định tuyến nguồn theo câu hỏi
  (query → document_type), không phải lọc/boost tĩnh.
