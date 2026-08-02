# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Khởi động (Warm-up) (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:*

**Ví dụ có độ tương tự CAO:**
- Câu A:
- Câu B:
- Tại sao tương đồng:

**Ví dụ có độ tương tự THẤP:**
- Câu A:
- Câu B:
- Tại sao khác:

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:*

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> *Đáp án:*

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:*

---

## 2. Lựa chọn tài liệu (Document Selection) — Nhóm (10 điểm)

### Chủ đề (Domain) & Lý Do Chọn

**Chủ đề:** [ví dụ: Customer support FAQ, Luật Việt Nam, công thức nấu ăn, ...]

**Tại sao nhóm chọn chủ đề này?**
> *Viết 2-3 câu:*

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [ ] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [ ] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chiến lược chia nhỏ (Chunking Strategy) — Cá nhân chọn, nhóm so sánh (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| | FixedSizeChunker (`fixed_size`) | | | |
| | SentenceChunker (`by_sentences`) | | | |
| | RecursiveChunker (`recursive`) | | | |

### Chiến Lược Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / chiến lược tùy chỉnh (custom)]

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: chiến lược của bạn chia nhỏ thế nào? Dựa trên dấu hiệu gì?*

**Tại sao tôi chọn chiến lược này cho chủ đề của nhóm?**
> *Viết 2-3 câu: chủ đề có đặc điểm gì (pattern) mà chiến lược này khai thác?*

**Đoạn mã (Code snippet) (nếu là tùy chỉnh):**
```python
# Dán mã nguồn (implementation) vào đây
```

### So Sánh: Chiến lược của tôi vs Đường cơ sở (Baseline)

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Chất lượng truy xuất? |
|-----------|----------|-------------|------------|--------------------|
| | baseline tốt nhất | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: dùng biểu thức chính quy (regex) gì để phát hiện câu? Xử lý trường hợp ngoại lệ (edge case) nào?*

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: thuật toán hoạt động thế nào? Base case (trường hợp cơ sở) là gì?*

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Viết 2-3 câu: lưu trữ thế nào? Tính độ tương tự ra sao?*

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: lọc (filter) trước hay sau? Xóa bằng cách nào?*

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: cấu trúc prompt? Cách đưa ngữ cảnh (inject context) vào thế nào?*

### Kết Quả Kiểm Thử (Test Results)

```
# Dán kết quả (output) của: pytest tests/ -v
```

**Số lượng bài test vượt qua (pass):** __ / __

---

## 5. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | cao / thấp | | |
| 2 | | | cao / thấp | | |
| 3 | | | cao / thấp | | |
| 4 | | | cao / thấp | | |
| 5 | | | cao / thấp | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Kết quả (Results) — Cá nhân (10 điểm)

Chạy 5 câu hỏi đánh giá của nhóm trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm.**

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** __ / 5

---

## 7. Những gì tôi học được (What I Learned) (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua buổi thuyết trình/demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Khởi động (Warm-up) | Cá nhân | / 5 |
| Lựa chọn tài liệu | Nhóm | / 10 |
| Chiến lược chia nhỏ (Chunking) | Nhóm | / 15 |
| Hướng tiếp cận của tôi | Cá nhân | / 10 |
| Dự đoán độ tương tự | Cá nhân | / 5 |
| Kết quả truy xuất | Cá nhân | / 10 |
| Hoàn thiện code (Core implementation - tests) | Cá nhân | / 30 |
| Thuyết trình (Demo) | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
