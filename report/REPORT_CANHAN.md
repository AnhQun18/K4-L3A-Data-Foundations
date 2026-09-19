# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Anh Quân  
**Nhóm:** G25  
**Ngày:** 19/09/2026  
**Lớp:** K4-L3A (Chủ đề: Dịch vụ & Quy chế Đào tạo Đại học)

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**  
Độ tương tự cosine cao nghĩa là hai vector chỉ về cùng một hướng trong không gian đa chiều, thể hiện rằng hai đoạn văn bản có ý nghĩa ngữ nghĩa (semantic meaning) rất giống nhau, bất kể độ dài thực tế hay số lượng từ vựng của chúng.

**Ví dụ có độ tương tự CAO:**
- Câu A: *"Sinh viên có thể đăng ký học phần trên trang tín chỉ."*
- Câu B: *"Việc ghi danh các môn học được thực hiện qua cổng thông tin tín chỉ của trường."*
- *Giải thích:* Mặc dù dùng từ vựng khác nhau ("đăng ký học phần" vs "ghi danh môn học"), ý nghĩa cốt lõi của hai câu hoàn toàn tương đồng về mặt hành động và địa điểm.

**Ví dụ có độ tương tự THẤP:**
- Câu A: *"Sinh viên có thể đăng ký học phần trên trang tín chỉ."*
- Câu B: *"Sinh viên bị cảnh cáo học tập nếu điểm trung bình tích lũy quá thấp."*
- *Giải thích:* Hai câu nói về hai nghiệp vụ hoàn toàn khác nhau (đăng ký môn học và xử lý kỷ luật/học vụ) mặc dù có chung từ khóa "Sinh viên".

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**  
Cosine similarity đo góc giữa hai vector (phản ánh phương hướng ngữ nghĩa) thay vì khoảng cách tuyệt đối giữa hai điểm cuối. Khoảng cách Euclid bị chi phối nặng nề bởi độ dài văn bản (văn bản dài hơn sẽ có vector độ lớn lớn hơn nếu chưa chuẩn hóa, làm tăng khoảng cách vô lý), trong khi ngữ nghĩa cốt lõi không thay đổi khi lặp lại câu từ. Khi các vector đã được chuẩn hóa L2 ($\|v\| = 1$), cosine similarity tương đương với tích vô hướng (dot product), tính toán cực kỳ nhanh chóng.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, `chunk_size=500`, `overlap=50`. Bao nhiêu chunks?**  
- Áp dụng công thức: $\text{Số chunks} = \left\lceil \frac{\text{Độ dài tài liệu} - \text{Độ chồng chéo}}{\text{Kích thước chunk} - \text{Độ chồng chéo}} \right\rceil$
- Bước nhảy (stride) = $500 - 50 = 450$ ký tự.
- Thay số: $\left\lceil \frac{10000 - 50}{450} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = \lceil 22.11 \rceil = 23\text{ chunks}$.
- **Đáp án:** 23 chunks. Đã kiểm chứng lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` cho ra đúng 23 chunks.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**  
- Bước nhảy mới: $500 - 100 = 400$ ký tự. Số chunk tăng lên thành $\lceil (10000 - 100) / 400 \rceil = \lceil 24.75 \rceil = 25\text{ chunks}$ (tăng thêm 2 chunks).
- Lý do tăng độ chồng chéo: Để bảo toàn liên kết ngữ nghĩa giữa các ranh giới cắt. Trong văn bản quy chế đại học, các điều kiện hoặc quy trình thường nối tiếp nhau qua 2-3 câu liên tục; overlap lớn hơn giúp ngăn tình trạng thông tin bị cắt đôi giữa chừng, đảm bảo mô hình LLM luôn nhận đủ ngữ cảnh trọn vẹn ở cả hai chunk kế cận.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Chiến lược chia nhỏ cá nhân được chọn (My Selected Chunking Strategy)

- **Chiến lược được chọn:** **Chia nhỏ đệ quy (Recursive Chunking)** với tham số tối ưu `chunk_size = 500`, `overlap = 50`.
- **Lý do chọn cho tập dữ liệu Quy chế Đại học (NEU):**
  Văn bản quy định học vụ tại trường Đại học Kinh tế Quốc dân được cấu trúc chặt chẽ theo thứ bậc: *Chương $\rightarrow$ Điều $\rightarrow$ Khoản $\rightarrow$ Điểm*. Các điều luật thường bao gồm một câu mở đầu nêu nguyên tắc (ví dụ: *"Sinh viên được xét tốt nghiệp khi thỏa mãn các điều kiện sau..."*) kèm theo danh sách các gạch đầu dòng liệt kê cụ thể. 
  - Nếu dùng `FixedSizeChunker`, thuật toán sẽ cắt cơ học tại ký tự thứ 500, dẫn đến việc câu mở đầu bị tách rời khỏi danh sách điều kiện, khiến chunk phía sau mất hoàn toàn chủ ngữ và ngữ cảnh pháp lý.
  - Nếu dùng `SentenceChunker`, do các quy định thường sử dụng dấu chấm phẩy (`;`) giữa các điểm hoặc câu ghép rất dài, việc tách thuần túy theo dấu chấm câu làm kích thước chunk biến thiên thất thường (quá dài hoặc quá vụn).
  - Do đó, `RecursiveChunker` là chiến lược tối ưu nhất: nó ưu tiên tách ở các ranh giới lớn (`\n\n` giữa các điều và `\n` giữa các khoản), đồng thời cơ chế greedy merge gom các dòng ngắn liền kề lại cho tới khi đạt kích thước ~500 ký tự. Kết quả là mỗi chunk lưu giữ trọn vẹn một quy định hoàn chỉnh.

- **So sánh với đường cơ sở (Baseline Analysis):**
  Chạy `ChunkingStrategyComparator().compare()` trên tài liệu mẫu `bo-quy-dinh-dao-tao.md`:
  - `FixedSizeChunker(200, 20)`: 428 chunks, độ dài trung bình 198 ký tự $\rightarrow$ Nhiều chunk bị cắt đôi giữa câu/từ.
  - `SentenceChunker(3)`: 312 chunks, độ dài biến thiên lớn (từ 80 đến 650 ký tự) $\rightarrow$ Ngữ cảnh không đồng đều.
  - `RecursiveChunker(500)`: 196 chunks, độ dài trung bình 412 ký tự $\rightarrow$ Giữ trọn vẹn ngữ nghĩa điều khoản, số lượng chunk vừa phải, tối ưu cho tìm kiếm vector.

### Chi tiết các hàm chia nhỏ (Chunking Implementation)

**`SentenceChunker.chunk`**:
Tôi sử dụng kỹ thuật positive lookbehind trong biểu thức chính quy `r'(?<=[.!?])\s+|(?<=\.)\n'` để tách câu ngay sau các dấu chấm, chấm hỏi, chấm than hoặc dấu chấm xuống dòng mà không làm mất (nuốt) dấu câu của văn bản gốc. Sau khi tách các câu riêng rẽ và dùng `.strip()` loại bỏ khoảng trắng dư thừa, các câu được gom nhóm tuần tự theo từng cụm `max_sentences_per_chunk` (mặc định là 3 câu/chunk). Trường hợp văn bản rỗng được kiểm tra đầu tiên để trả về ngay `[]`.

**`RecursiveChunker.chunk` / `_split`**:
Thuật toán hoạt động theo nguyên lý đệ quy 2 chiều:
1. *Ưu tiên ranh giới lớn:* Thử các ký tự phân cách theo thứ tự giảm dần về kích thước ngữ nghĩa `["\n\n", "\n", ". ", " ", ""]`.
2. *Gom cụm liền kề (Greedy merge):* Sau khi tách theo separator hiện tại, các đoạn nhỏ được cộng dồn lại với nhau cho đến khi gần chạm ngưỡng `chunk_size` mới tạo chunk mới.
3. *Đệ quy xuống sâu:* Nếu có một đoạn đơn lẻ vẫn vượt quá `chunk_size`, hàm gọi đệ quy tiếp với danh sách separators còn lại (`next_seps`).
4. *Base cases:* Nếu đoạn văn bản đã $\le \text{chunk\_size}$, trả về `[current_text]`. Nếu đã hết separators (`remaining_separators == []`), thực hiện cắt cứng theo từng lát `chunk_size` ký tự để tránh rơi vào vòng lặp vô hạn.

### Lớp EmbeddingStore

**`add_documents` + `search`**:
Store hỗ trợ cả lưu trữ trong bộ nhớ (In-memory) và ChromaDB (nếu môi trường có sẵn). Mặc định, dữ liệu được quản lý in-memory dạng danh sách các bản ghi chuẩn hóa `dict` chứa `id`, `content`, `metadata`, và `embedding`. Khi tìm kiếm `search()`, vector câu hỏi được nhúng qua `self._embedding_fn(query)`, sau đó tính tích vô hướng (dot product) với toàn bộ vector đã lưu trữ bằng hàm trợ giúp `_dot()`. Vì các vector embedding đã được chuẩn hóa đơn vị ($\|v\| = 1$), dot product phản ánh trực tiếp Cosine Similarity. Kết quả được sắp xếp giảm dần theo điểm `score` và trích xuất `top_k` phần tử.

**`search_with_filter` + `delete_document`**:
- **Pre-filtering (Lọc trước):** Trong `search_with_filter`, bộ lọc metadata được áp dụng *TRƯỚC* khi thực hiện so khớp vector similarity. Điều này tối ưu hiệu năng tính toán và quan trọng nhất là bảo đảm $k$ kết quả trả về luôn thỏa mãn điều kiện lọc, tránh trường hợp post-filtering làm mất sạch kết quả do top-k bị các tài liệu ngoài phạm vi chiếm chỗ. Cả `search()` và `search_with_filter()` đều tái sử dụng chung phương thức `_search_records()` để bảo đảm tính nhất quán.
- **Xóa tài liệu (`delete_document`):** Duyệt qua store và loại bỏ tất cả các chunk có `id == doc_id` hoặc có `metadata.get('doc_id') == doc_id`. Trả về `True` nếu số lượng phần tử giảm đi, ngược lại trả về `False`.

### Tác tử KnowledgeBaseAgent

**`answer`**:
Thực hiện mô hình RAG tiêu chuẩn gồm 3 bước:
1. Gọi `store.search(question, top_k=top_k)` để trích xuất các đoạn văn bản có độ tương đồng cao nhất.
2. Dựng prompt rõ ràng phân định ranh giới giữa ngữ cảnh và câu hỏi: `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"`.
3. Chuyển prompt hoàn chỉnh cho hàm tạo sinh `llm_fn` để tổng hợp câu trả lời dựa trên sự kiện được cung cấp, ngăn ngừa ảo giác (hallucination).

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Đã chạy kiểm thử toàn diện và vượt qua tất cả **42/42 tests**:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\admin\Documents\Lab\K4-L3A-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\admin\Documents\Lab\K4-L3A-Data-Foundations
plugins: anyio-4.15.1
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.16s ==============================
```

**Số lượng bài test vượt qua:** **42 / 42** (100%)

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Đo lường thực tế bằng hàm `compute_similarity` kết hợp mô hình `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`:

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|:---:|:---|:---|:---:|:---:|:---:|
| 1 | "Quy chế đào tạo đại học NEU" | "Quy định học vụ dành cho sinh viên KTQD" | Cao | 0.84 | Có |
| 2 | "Đăng ký tín chỉ trực tuyến" | "Hướng dẫn nộp học phí qua Momo" | Thấp | 0.38 | Có |
| 3 | "Hủy lớp học phần" | "Rút bớt học phần đã đăng ký" | Cao | 0.79 | Có |
| 4 | "Điểm trung bình tích lũy (CPA)" | "Thang điểm đánh giá học phần" | Cao | 0.73 | Có |
| 5 | "Sinh viên K63 trở đi" | "Quy định đối với khóa 63 và các khóa sau" | Rất cao | 0.92 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**  
Kết quả ấn tượng nhất là Cặp 3 ("Hủy lớp học phần" vs "Rút bớt học phần đã đăng ký") đạt độ tương đồng xấp xỉ 0.79 mặc dù không trùng nhau các từ khóa cốt lõi ("Hủy" vs "Rút bớt"). Điều này chứng minh rằng Embedding đa ngữ không chỉ so khớp từ vựng đơn thuần (lexical matching), mà nó chiếu các khái niệm đồng nghĩa vào các vùng vector rất gần nhau trong không gian biểu diễn ngữ nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy thực nghiệm đánh giá tự động qua công cụ `bench.py` với chiến lược cá nhân `RecursiveChunker(chunk_size=500, overlap=50)` trên toàn bộ 7 tài liệu quy chế NEU (1.414 chunks), sử dụng mô hình embedding đa ngữ `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

Chi tiết log đầy đủ được lưu tại `ket_qua_benchmark.txt`.

### Bảng Kết Quả Đánh Giá 5 Benchmark Queries

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|---|---|:---:|:---:|---|
| 1 | Trường tổ chức cho sinh viên đăng ký học muộn nhất bao lâu trước khi bắt đầu học kỳ? | *"3. Thời gian đăng ký: Trường tổ chức cho sinh viên đăng ký học muộn nhất 3 tuần trước thời điểm bắt đầu học kỳ..."* (Nguồn: `quy-dinh-ve-dao-tao-theo-chuong-trinh-tien-tien`) | 0.8596 | **Có (Top-1)** | Trường tổ chức cho sinh viên đăng ký học muộn nhất 3 tuần trước khi bắt đầu học kỳ (đối với CTTT/CLC) hoặc muộn nhất 2 tuần (theo Quy chế K63). |
| 2 | Học cải thiện điểm được tối đa bao nhiêu tín chỉ trong học kỳ 1? | *"Sinh viên chỉ được học cải thiện điểm không quá 8 tín chỉ đối với học kỳ 1, học kỳ 2; không quá 5 tín chỉ đối với học kỳ hè..."* (Nguồn: `nhung-dieu-sv-dh-ktqd-can-biet--2014`) | 0.7974 | **Có (Top-1)** | Sinh viên chỉ được học cải thiện điểm tối đa không quá 8 tín chỉ trong học kỳ 1 (và học kỳ 2), và không quá 5 tín chỉ đối với học kỳ hè. |
| 3 | Khi không đồng ý với điểm thi thì làm gì? | *- Top 1 (0.6048):* Trừ điểm khi vi phạm quy chế thi (`nhung-dieu-sv-dh-ktqd-can-biet--2014`).<br>*- Top 3 (0.5883):* *"Chậm nhất 1 tuần sau khi công bố điểm... làm đơn đề nghị xem lại bài thi..."* (`quy-dinh-dao-tao-chat-luong-cao`) | 0.6048 *(Top 3: 0.5883)* | **Có (Top-3)** *(Failure Case phân tích dưới)* | Sinh viên làm đơn đề nghị xem lại kết quả bài thi học phần (phúc khảo) trong thời hạn quy định sau khi công bố kết quả thi. |
| 4 | Sinh viên được tuyển chọn vào chương trình Chất lượng cao như thế nào? | *"Sau khi đăng ký dự thi vào Chương trình Chất lượng cao nếu đạt số điểm trúng tuyển nhất định (tùy theo từng năm) mới được nhập học... Hiệu trưởng xét tuyển bổ sung vào đầu năm thứ 2..."* (Nguồn: `quy-dinh-dao-tao-chat-luong-cao`) | 0.7518 | **Có (Top-1)** | Sinh viên đã trúng tuyển đại học chính quy nộp hồ sơ xét tuyển/dự thi vào CT Chất lượng cao và đạt điểm trúng tuyển; có thể được xét tuyển bổ sung vào đầu năm thứ 2. |
| 5 | Điều kiện để được xét công nhận tốt nghiệp gồm những gì? | *"Căn cứ đề nghị của Hội đồng xét tốt nghiệp, Hiệu trưởng ra quyết định công nhận tốt nghiệp và cấp bằng tốt nghiệp... tích lũy đủ tín chỉ, CPA đạt từ trung bình, chứng chỉ GDQP-GDTC..."* (Nguồn: `1155-quyche-daotao-daihoc-k63-tr-e1-bb-9f--c4-91i`) | 0.7957 | **Có (Top-1)** | Sinh viên được công nhận tốt nghiệp khi tích lũy đủ số tín chỉ, đạt chuẩn đầu ra ngoại ngữ/tin học, CPA toàn khóa từ 2.0 trở lên, có chứng chỉ GDQP-AN, GDTC và điểm rèn luyện đạt yêu cầu. |

- **Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3:** **5 / 5** (100%).
- **Tổng điểm chất lượng truy xuất:** **9 / 10 điểm** (theo thang điểm tại `docs/SCORING.md`: 4 câu đạt 2 điểm do Top-1 chứa đáp án, 1 câu đạt 1 điểm do đáp án nằm ở Top-3).

---

### Phân tích A/B Testing (Lọc Metadata vs Không Lọc) trên Câu hỏi 1

Khi chạy thử nghiệm A/B trên Câu 1:
- **Không lọc metadata:** Cả 3 kết quả hàng đầu đều lấy từ các tài liệu sinh viên do câu từ trùng khớp cao (`score = 0.8596`).
- **Có lọc `metadata_filter={"audience": "student"}`:** Hệ thống loại bỏ hoàn toàn các tài liệu quy chế mang tính định hướng cơ quan quản lý (ví dụ: `thong-tu-08-2021-tt-bgddt...` có `audience: all`), bảo đảm 100% tài liệu được truy xuất là các hướng dẫn và thời khóa biểu áp dụng trực tiếp cho người học.

---

### Phân tích Trường hợp Lỗi (Failure Analysis — Bài tập 3.5)

- **Hiện tượng:** Ở Câu 3 (*"Khi không đồng ý với điểm thi thì làm gì?"*), chunk chứa đáp án chuẩn về thủ tục phúc khảo điểm chỉ xếp vị trí **Top-3** (score: 0.5883), trong khi **Top-1** (score: 0.6048) lại là đoạn nói về xử lý kỷ luật sinh viên khi vi phạm phòng thi.
- **Nguyên nhân:**
  1. *Độ tương đồng từ khóa chung:* Các cụm từ "điểm thi", "khi thi", "học phần" xuất hiện dày đặc trong cả quy chế kỷ luật thi và quy chế phúc khảo, khiến Cosine Similarity giữa câu hỏi và đoạn kỷ luật vô tình cao hơn.
  2. *Thiếu từ khóa chuyên biệt trong câu hỏi:* Người hỏi dùng từ thông thường "không đồng ý với điểm thi" thay vì thuật ngữ học thuật "phúc khảo" hoặc "xem lại kết quả bài thi".
- **Đề xuất cải thiện:**
  - Bổ sung kỹ thuật **Hybrid Search** (kết hợp Dense Vector Search với BM25 Keyword Search) để tăng trọng số các từ khóa thủ tục như "đơn đề nghị", "thanh tra".
  - Giảm `chunk_size` từ 500 xuống 300 đối với các mục thủ tục hành chính, hoặc trích xuất chunk theo cấu trúc tiêu đề điều khoản (`HeadingChunker`) để tên điều luật *"Điều... Phúc khảo và giải quyết khiếu nại"* được gắn kèm vào nội dung.

---

### Điều hay nhất tôi học được từ thành viên khác / nhóm khác

Chiến lược chia nhỏ theo câu (`SentenceChunker`) hoạt động rất tốt đối với các câu hỏi ngắn mang tính định nghĩa, nhưng lại dễ bị mất liên kết nếu nội dung một điều luật kéo dài qua nhiều đoạn. Ngược lại, chia nhỏ đệ quy (`RecursiveChunker`) với `chunk_size=500` và `overlap=50` tạo ra sự cân bằng tốt nhất: vừa bảo toàn được cấu trúc tự nhiên của văn bản quy định, vừa kiểm soát được kích thước chunk trong ngưỡng tối ưu của mô hình embedding.

---

## 6. Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tối đa | Điểm tự đánh giá | Minh chứng / Ghi chú |
|:---|:---:|:---:|:---|
| **Khởi động (Warm-up)** | 5 | **5 / 5** | Giải thích Cosine vs Euclidean rõ ràng, giải toán chunking chính xác (23 và 25 chunks). |
| **Hướng tiếp cận của tôi (My Approach)** | 10 | **10 / 10** | Trình bày cặn kẽ thuật toán `SentenceChunker`, `RecursiveChunker`, pre-filtering trong `EmbeddingStore`, RAG prompt. |
| **Hoàn thiện code (Core Implementation)** | 30 | **30 / 30** | Vượt qua 42/42 tests (`pytest tests/ -v`). |
| **Dự đoán độ tương tự (Similarity Predictions)** | 5 | **5 / 5** | Có bảng 5 cặp câu, đo bằng `LocalEmbedder` và phân tích bất ngờ. |
| **Kết quả truy xuất của tôi (Competition Results)** | 10 | **10 / 10** | Đạt 9/10 điểm retrieval trên 5 câu hỏi benchmark thực tế, có phân tích A/B và failure case. |
| **TỔNG PHẦN CÁ NHÂN** | **60** | **60 / 60** | **Hoàn thành xuất sắc toàn bộ yêu cầu phần cá nhân.** |
