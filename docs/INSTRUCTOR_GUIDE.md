# Hướng dẫn Giảng viên (Instructor Guide): Lab 7 - Nền tảng Dữ liệu (Data Foundations): Embedding & Vector Store

Hướng dẫn này dành cho giảng viên để dẫn dắt buổi lab 4.5 giờ. Lab chia làm 2 giai đoạn (phase): **cá nhân** (lập trình) và **nhóm** (so sánh chiến lược, học hỏi lẫn nhau).

> Chuẩn môi trường: Python **3.11.x**. Phần lõi (core) chỉ dùng `requirements.txt`; không yêu cầu PyTorch, API key hay GPU.

---

## Mục Tiêu Học Tập Cốt Lõi

1. **Hiểu về Embedding (Embedding Intuition) (G2)**: Hiểu độ tương tự cosine (cosine similarity), dự đoán được điểm tương đồng, nhận ra giới hạn của embedding.
2. **Các thao tác trên Vector Store (G3)**: Triển khai các chức năng lưu trữ (store) / tìm kiếm (search) / lọc (filter) / xóa (delete); giải thích được khi nào việc lọc bằng metadata (metadata filtering) giúp ích hoặc gây hại.
3. **Quy trình hoàn chỉnh (Full Pipeline) (G4)**: Triển khai từng bước Document → Chunk → Embed → Store → Query → Inject; so sánh các chiến lược chia nhỏ (chunking strategies).
4. **Chiến lược dữ liệu (Data Strategy) (G5)**: Chọn dữ liệu, thiết kế metadata, tối ưu chunking — hiểu rằng chất lượng dữ liệu (data quality) quan trọng hơn việc chọn mô hình.

---

## Ghi Chú Cho Giảng Viên: Embedder Thật Là Tùy Chọn

- Lab này **không bắt buộc** sinh viên cài embedder thật.
- Luồng mặc định cho lớp học vẫn là trình nhúng giả lập `_mock_embed`, nên sinh viên vẫn có thể hoàn thành lab và vượt qua bài kiểm thử (pass test) mà không cần tải mô hình nào.
- Nếu sinh viên muốn thử embedding thật trên máy cá nhân, gói `src` đã hỗ trợ cả:
  - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` qua thư viện `sentence-transformers` (tùy chọn; phù hợp kho ngữ liệu tiếng Việt)
  - OpenAI embeddings qua thư viện `openai`

Ví dụ về trình nhúng cục bộ (local embedder):

```bash
pip install -r requirements-local.txt
python3 - <<'PY'
from src import LocalEmbedder
embedder = LocalEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

Ví dụ về OpenAI embedder:

```bash
pip install openai
export OPENAI_API_KEY=your-key-here
python3 - <<'PY'
from src import OpenAIEmbedder
embedder = OpenAIEmbedder()
print(embedder._backend_name)
print(len(embedder("embedding smoke test")))
PY
```

- Khuyến nghị giảng viên nói rõ ngay từ đầu: **“Local/OpenAI embedder là điểm cộng (bonus) / tùy chọn (optional), không phải điều kiện để hoàn thành lab.”**
- Khi có sinh viên máy yếu, mạng chậm, không có API key, hoặc không muốn tải mô hình, hãy hướng họ tiếp tục với `_mock_embed` để tránh bị kẹt ở phần thiết lập (setup).
- Thư mục `src_w_solution/` là bài giải tham khảo (reference solution) dành cho giảng viên / người bảo trì (maintainer). Không phân phối thư mục này cho sinh viên.

---

## Tiến trình (Timeline) & Luồng hoạt động (Flow) (4.5 giờ)

### Giai đoạn 1: Chuẩn bị tài liệu (Document Preparation) (30 phút, 0:00–0:30)

**Hoạt động (nhóm):**
- Nhóm chọn chủ đề (domain) (FAQ, luật, công thức nấu ăn, y tế, tài liệu kỹ thuật, v.v.)
- Thu thập 5-10 tài liệu, chuyển sang định dạng `.txt`/`.md`, đặt vào thư mục `data/`
- Ghi lại `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực); không dùng dữ liệu cá nhân hoặc tài liệu không được phép chia sẻ
- Thiết kế cấu trúc metadata (ít nhất 2 trường hữu ích cho truy xuất)

**Vai trò giảng viên:**
- Giải thích cấu trúc lab: "30 phút chuẩn bị tài liệu nhóm → mỗi người tự code → mỗi người thử chiến lược riêng → so sánh trong nhóm → thuyết trình (demo) với lớp"
- Gợi ý chủ đề nếu nhóm chưa quyết định được
- Nhấn mạnh: "Chọn tài liệu có cấu trúc rõ ràng — chất lượng tài liệu quyết định kết quả"

### Giai đoạn 2: Lập trình cá nhân (Individual Coding) (90 phút, 0:30–2:00)

**Khởi động (Warm-up) (10 phút):**
- Bài 1.1: Độ tương tự Cosine — giải thích bằng ngôn ngữ tự nhiên
- Bài 1.2: Bài toán Chunking — tính toán số lượng chunk

**Thực hành lập trình (Implementation) (80 phút):**
- Mỗi sinh viên **tự mình** lập trình tất cả các phần CẦN LÀM (TODO) trong `src/chunking.py`, `src/store.py`, và `src/agent.py`
- Lớp `Document` và `FixedSizeChunker` đã được lập trình sẵn làm ví dụ
- Thứ tự gợi ý: `SentenceChunker` → `RecursiveChunker` → `compute_similarity` → `ChunkingStrategyComparator` → `EmbeddingStore` → `KnowledgeBaseAgent`

**Vai trò giảng viên:**
- **Nhấn mạnh**: "Đây là phần cá nhân — mỗi người tự code"
- **Điểm kiểm tra 1 (Checkpoint 1) (1:00)**: "Ai đã vượt qua phần chunking (`TestSentenceChunker`, `TestRecursiveChunker`)?" — giải thích lại nếu < 50% lớp làm được
- **Điểm kiểm tra 2 (Checkpoint 2) (1:30)**: "Ai đã vượt qua TestEmbeddingStore?" — hỗ trợ sửa lỗi (debug) nếu cần

### Giai đoạn 3: Thiết kế chiến lược (Strategy Design) (60 phút, 2:00–3:00)

**Hoạt động:**
- Nhóm thống nhất **5 câu hỏi đánh giá (benchmark queries) + câu trả lời chuẩn (gold answers)**
- Mỗi thành viên **chọn chiến lược riêng** (phương pháp chunking, tham số, cấu trúc metadata)
- Chạy đường cơ sở (baseline) để so sánh, thiết kế chiến lược tùy chỉnh (custom strategy) nếu muốn
- Đưa (Index) tài liệu vào EmbeddingStore với chiến lược riêng

**Vai trò giảng viên:**
- Khuyến khích mỗi người thử chiến lược khác nhau: "Một người thử `FixedSizeChunker`, một người thử `RecursiveChunker`, một người thử custom"
- Kiểm tra các câu hỏi đánh giá: "Các câu hỏi có đủ đa dạng không?"
- Nhắc nhở: câu trả lời chuẩn phải cụ thể, có thể kiểm chứng (verifiable)

**Điểm kiểm tra (Checkpoint) (2:45):** Mỗi nhóm phải có sẵn 5 câu hỏi đánh giá + câu trả lời chuẩn

### Giai đoạn 4: So Sánh & Thảo Luận Trong Nhóm (30 phút, 3:00–3:30)

**Hoạt động:**
1. Mỗi thành viên chạy 5 câu hỏi đánh giá với chiến lược riêng (10 phút)
2. So sánh kết quả trong nhóm (10 phút):
   - Chiến lược nào tốt nhất? Tại sao?
   - Có câu hỏi nào chiến lược A thắng nhưng B thua?
3. Chuẩn bị thuyết trình (demo) (10 phút): chọn những phân tích (insights) hay nhất để chia sẻ

**Vai trò giảng viên:**
- Đi quanh lớp, đặt câu hỏi: "Chiến lược nào thắng? Các bạn có giải thích được tại sao không?"
- Thu thập 2-3 phát hiện hay từ các nhóm để sử dụng trong phần thảo luận chung

### Giai đoạn 5: Thuyết trình (Demo) & Thảo Luận Liên Nhóm (60 phút, 3:30–4:30)

**Định dạng thuyết trình (8-10 phút/nhóm):**
1. Giới thiệu chủ đề (domain) + bộ tài liệu (1 phút)
2. Mỗi thành viên tóm tắt chiến lược của mình (2 phút)
3. So sánh: chiến lược nào thắng trên bộ dữ liệu này? Tại sao? (3 phút)
4. Demo 1-2 câu hỏi trực tiếp (live) (2 phút)
5. Hỏi đáp (Q&A) từ các nhóm khác + giảng viên (2 phút)

**Câu hỏi gợi ý cho phần thảo luận:**
- "Nếu chuyển sang chủ đề khác, chiến lược nào vẫn hoạt động tốt?"
- "Việc lọc bằng Metadata giúp ích ở đâu? Ở đâu nó làm mất đi kết quả tốt?"
- "Từ kết quả của nhóm bạn, nhóm mình có thể áp dụng được bài học gì?"

**Đúc kết của giảng viên (Wrap-up) (5 phút):**
- Bài học cốt lõi: "Cùng tài liệu, nhưng chiến lược khác nhau → kết quả rất khác nhau. Hiểu rõ tại sao lại quan trọng hơn là chỉ chạy được code."
- Nhắc nhở: mỗi sinh viên nộp 1 bản báo cáo (phần làm việc nhóm giống nhau, phần cá nhân + chiến lược khác nhau)
- Kết nối với Ngày 8 (Quy trình RAG hoàn chỉnh)

---

## Sai Lầm Phổ Biến

| Sai lầm | Cách xử lý |
|---------|------------|
| **Độ chồng chéo (overlap) > kích thước chunk (chunk_size)** | Hỏi: "bước nhảy (step) = chunk_size - overlap. Nếu overlap >= chunk_size thì bước nhảy là gì?" |
| **Quên chuẩn hóa (normalize) vector** trong compute_similarity | Chỉ ra công thức: cần chia cho \|\|a\|\| * \|\|b\|\| |
| **search_with_filter không lọc trước** | Sinh viên tìm kiếm (search) rồi mới lọc (filter) → kết quả sai. Phải lọc trước, rồi mới tìm kiếm |
| **KnowledgeBaseAgent không đưa ngữ cảnh (inject context) vào** | Kiểm tra: câu lệnh (prompt) có chứa các chunk truy xuất được không? |
| **Tất cả thành viên chọn cùng một chiến lược** | Yêu cầu mỗi người thử chiến lược khác nhau — mục tiêu là để so sánh |
| **Câu hỏi đánh giá (Benchmark queries) quá dễ hoặc giống nhau** | Yêu cầu đa dạng: câu hỏi thực tế (factual), yêu cầu thông tin từ nhiều chunk, hoặc phụ thuộc metadata |

---

## Tiêu Chí Thành Công

Buổi lab thành công nếu:
- Mọi sinh viên vượt qua (pass) được ít nhất 70% bài kiểm thử cá nhân
- Mỗi nhóm có ít nhất 2 chiến lược khác nhau để so sánh
- Sinh viên giải thích được tại sao chiến lược A tốt hơn B trên dữ liệu cụ thể
- Buổi thuyết trình có sự thảo luận sôi nổi giữa các nhóm
- Sinh viên kết nối được: chiến lược dữ liệu ảnh hưởng trực tiếp đến chất lượng truy xuất (retrieval quality)

---

*"Chất lượng dữ liệu thường quan trọng hơn việc đổi sang mô hình đắt tiền hơn. Hãy dạy sinh viên nhìn vào dữ liệu trước khi nhìn vào mô hình."*
