# K3 Variant — University Services Retrieval

K3 dùng cùng core coding contract với K4, nhưng Phase 2 phải xây dựng knowledge base về **dịch vụ hoặc quy định đại học** (ví dụ: đăng ký môn, học phí, học bổng, thư viện, ký túc xá).

## Quy tắc riêng của K3

- Mỗi document phải có metadata `audience` (ví dụ: `student`, `faculty`, `staff`) và ít nhất một field hữu ích khác.
- Ngoài metadata retrieval, mỗi document phải có `source_url`, `retrieved_at` và `document_version`; chỉ dùng quy định/dịch vụ công khai hoặc được phép chia sẻ.
- Trong 5 benchmark query, có ít nhất một query cần `metadata_filter={"audience": "student"}` để tránh lấy tài liệu dành cho đối tượng khác.
- Ít nhất một thành viên thử chunking theo heading/section của handbook hoặc quy định học vụ.
- Gold answer phải trích được từ tài liệu nhóm thu thập, không suy đoán quy định của trường.

Thư mục `data/k3_university/` có dữ liệu khởi động nhỏ; nhóm vẫn cần bổ sung corpus 5–10 document theo yêu cầu Lab.
