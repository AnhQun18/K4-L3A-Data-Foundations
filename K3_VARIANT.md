# Biến thể K3 (K3 Variant) — Truy xuất Dịch vụ Đại học (University Services Retrieval)

K3 dùng chung cam kết mã nguồn cốt lõi (core coding contract) với K4, nhưng Giai đoạn 2 (Phase 2) phải xây dựng cơ sở tri thức (knowledge base) về **dịch vụ hoặc quy định đại học** (ví dụ: đăng ký môn, học phí, học bổng, thư viện, ký túc xá).

## Quy tắc riêng của K3

- Mỗi tài liệu (document) phải có metadata `audience` (ví dụ: `student`, `faculty`, `staff`) và ít nhất một trường (field) hữu ích khác.
- Ngoài việc truy xuất bằng metadata, mỗi tài liệu phải có `source_url`, `retrieved_at` và `document_version`; chỉ dùng quy định/dịch vụ công khai hoặc được phép chia sẻ.
- Trong 5 câu hỏi đánh giá (benchmark query), có ít nhất một câu hỏi cần `metadata_filter={"audience": "student"}` để tránh lấy tài liệu dành cho đối tượng khác.
- Ít nhất một thành viên thử chia nhỏ (chunking) theo tiêu đề/mục (heading/section) của sổ tay hoặc quy định học vụ.
- Câu trả lời chuẩn (Gold answer) phải trích được từ tài liệu nhóm thu thập, không suy đoán quy định của trường.

Thư mục `data/k3_university/` có dữ liệu khởi động nhỏ; nhóm vẫn cần bổ sung tập tài liệu (corpus) 5–10 tài liệu theo yêu cầu Lab.
