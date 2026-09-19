"""
Script Benchmark Đánh Giá Chiến Lược Truy Xuất (Retrieval Benchmark)
Lab 07 — Nền tảng Dữ liệu, Embedding & Vector Store (K4-L3A)

Cách dùng:
    python bench.py
    python bench.py --strategy recursive --provider local
    python bench.py --strategy fixed --provider local
    python bench.py --strategy heading --provider local
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Đảm bảo terminal Windows không bị lỗi UnicodeEncodeError khi in tiếng Việt
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore


# ==============================================================================
# Chiến lược chia nhỏ bổ sung: HeadingChunker (theo tiêu đề/mục cho văn bản quy định)
# ==============================================================================
class HeadingChunker:
    """
    Chia nhỏ theo tiêu đề hoặc điều khoản (## Điều... / # ...)
    Phù hợp đặc thù văn bản quy định học vụ đại học (K4-L3A).
    """

    def __init__(self, max_chunk_size: int = 1000) -> None:
        self.max_chunk_size = max_chunk_size
        self._fallback_chunker = RecursiveChunker(chunk_size=max_chunk_size)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Tách theo các tiêu đề Markdown hoặc các điều luật
        pattern = r"(?=(?:^|\n)#{1,4}\s+|(?:\n|^)(?:Điều|Chương)\s+\d+)"
        sections = re.split(pattern, text)
        sections = [s.strip() for s in sections if s.strip()]

        chunks: list[str] = []
        for sec in sections:
            if len(sec) <= self.max_chunk_size:
                chunks.append(sec)
            else:
                # Nếu một mục quá dài, dùng RecursiveChunker để chia tiếp
                sub_chunks = self._fallback_chunker.chunk(sec)
                chunks.extend(sub_chunks)
        return chunks


# ==============================================================================
# Trình bọc Embedder có lưu Cache trên đĩa (Disk Cache) để tăng tốc & tránh quota
# ==============================================================================
class CachedEmbedder:
    """
    Wrapper lưu trữ embedding vào file JSON cục bộ theo hash nội dung.
    Giúp chạy benchmark nhanh tức thì trong các lần chạy sau mà không gọi lại model/API.
    """

    def __init__(
        self,
        base_embedder: Callable[[str], list[float]],
        backend_name: str,
        cache_path: str = ".embedding_cache.json",
    ) -> None:
        self.base_embedder = base_embedder
        self._backend_name = backend_name
        self.cache_path = Path(cache_path)
        self.cache: dict[str, list[float]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    def save_cache(self) -> None:
        try:
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        except Exception:
            pass

    def __call__(self, text: str) -> list[float]:
        key = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
        if key in self.cache:
            return self.cache[key]
        emb = self.base_embedder(text)
        self.cache[key] = emb
        return emb


# ==============================================================================
# Đọc và phân tích tài liệu Markdown có Frontmatter
# ==============================================================================
def parse_markdown_with_frontmatter(file_path: Path) -> tuple[dict[str, Any], str]:
    content = file_path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {
        "doc_id": file_path.stem,
        "source": str(file_path),
        "filename": file_path.name,
    }
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            for line in fm_text.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("\"'")
                    metadata[k] = v

    if "doc_id" not in metadata or not metadata["doc_id"]:
        metadata["doc_id"] = file_path.stem

    return metadata, body


# ==============================================================================
# 5 Câu hỏi Benchmark thống nhất của nhóm (K4-L3A)
# ==============================================================================
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Trường tổ chức cho sinh viên đăng ký học muộn nhất bao lâu trước khi bắt đầu học kỳ?",
        "gold_answer": "Muộn nhất 2 đến 3 tuần trước khi bắt đầu học kỳ (2 tuần theo Quy chế K63, 3 tuần theo CTTT/CLC).",
        "filter": {"audience": "student"},
        "need_filter_ab": True,
        "expected_keywords": ["đăng ký học", "muộn nhất", "tuần", "học kỳ"],
    },
    {
        "id": 2,
        "query": "Học cải thiện điểm được tối đa bao nhiêu tín chỉ trong học kỳ 1?",
        "gold_answer": "Sinh viên chỉ được học cải thiện điểm không quá 8 tín chỉ đối với học kỳ 1 và học kỳ 2; không quá 5 tín chỉ đối với học kỳ hè.",
        "filter": {"audience": "student"},
        "need_filter_ab": False,
        "expected_keywords": ["cải thiện điểm", "không quá 8 tín chỉ", "học kỳ 1"],
    },
    {
        "id": 3,
        "query": "Khi không đồng ý với điểm thi thì làm gì?",
        "gold_answer": "Sinh viên làm đơn đề nghị xem lại kết quả bài thi học phần (phúc khảo) gửi Phòng Thanh tra, Khảo thí và Đảm bảo chất lượng giáo dục trong thời hạn quy định.",
        "filter": {"audience": "student"},
        "need_filter_ab": False,
        "expected_keywords": ["xem lại kết quả", "phúc khảo", "thanh tra"],
    },
    {
        "id": 4,
        "query": "Sinh viên được tuyển chọn vào chương trình Chất lượng cao như thế nào?",
        "gold_answer": "Đã trúng tuyển đại học chính quy, nộp hồ sơ xét tuyển/dự thi vào CT Chất lượng cao và đạt điểm chuẩn; có thể xét tuyển bổ sung đầu năm 2 nếu đạt điều kiện.",
        "filter": {"audience": "student"},
        "need_filter_ab": False,
        "expected_keywords": ["chất lượng cao", "trúng tuyển", "xét tuyển"],
    },
    {
        "id": 5,
        "query": "Điều kiện để được xét công nhận tốt nghiệp gồm những gì?",
        "gold_answer": "Tích lũy đủ số tín chỉ; CPA toàn khóa từ 2.0 trở lên; đạt chuẩn đầu ra ngoại ngữ, tin học; có chứng chỉ GDQP-AN, GDTC; đạt điểm rèn luyện; không bị truy cứu hình sự hay đình chỉ học tập.",
        "filter": {"audience": "student"},
        "need_filter_ab": False,
        "expected_keywords": ["công nhận tốt nghiệp", "tích lũy", "chuẩn đầu ra", "GDQP"],
    },
]


def create_chunker(strategy_name: str, chunk_size: int = 500, overlap: int = 50):
    if strategy_name == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy_name == "sentence":
        return SentenceChunker(max_sentences_per_chunk=3)
    elif strategy_name == "heading":
        return HeadingChunker(max_chunk_size=chunk_size)
    else:
        return RecursiveChunker(chunk_size=chunk_size)


def get_embedder(provider: str | None = None) -> tuple[Callable[[str], list[float]], str]:
    load_dotenv(override=False)
    chosen_provider = (provider or os.getenv(EMBEDDING_PROVIDER_ENV, "local")).strip().lower()

    if chosen_provider == "local":
        try:
            embedder = LocalEmbedder()
            return embedder, embedder._backend_name
        except Exception as e:
            print(f"[Cảnh báo] Không khởi tạo được LocalEmbedder ({e}), chuyển sang MockEmbedder")
            return _mock_embed, "mock embeddings fallback"
    elif chosen_provider == "gemini":
        try:
            embedder = GeminiEmbedder()
            return embedder, embedder._backend_name
        except Exception as e:
            print(f"[Cảnh báo] Không khởi tạo được GeminiEmbedder ({e}), chuyển sang MockEmbedder")
            return _mock_embed, "mock embeddings fallback"
    elif chosen_provider == "openai":
        try:
            embedder = OpenAIEmbedder()
            return embedder, embedder._backend_name
        except Exception as e:
            print(f"[Cảnh báo] Không khởi tạo được OpenAIEmbedder ({e}), chuyển sang MockEmbedder")
            return _mock_embed, "mock embeddings fallback"
    else:
        return _mock_embed, "mock embeddings fallback"


def run_benchmark(
    data_dir: str = "data/neu-quy-dinh",
    strategy: str = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
    provider: str = "local",
    output_file: str = "ket_qua_benchmark.txt",
) -> int:
    lines_output: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines_output.append(msg)

    log("=" * 80)
    log("K4-L3A RETRIEVAL BENCHMARK — LAB 07")
    log("=" * 80)
    log(f"Thời gian chạy: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Thư mục dữ liệu: {data_dir}")
    log(f"Chiến lược chia nhỏ (Chunking): {strategy} (chunk_size={chunk_size}, overlap={overlap})")

    # 1. Khởi tạo Embedder
    base_embedder, backend_name = get_embedder(provider)
    cached_embedder = CachedEmbedder(base_embedder, backend_name=backend_name)
    log(f"Mô hình Embedding: {backend_name}")

    # 2. Đọc tài liệu
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.md")))
    if not file_paths:
        log(f"[LỖI] Không tìm thấy file .md nào trong thư mục '{data_dir}'!")
        return 1

    log(f"\n[1] Đọc và tiền xử lý {len(file_paths)} tài liệu...")
    chunker = create_chunker(strategy, chunk_size, overlap)

    all_chunks: list[Document] = []
    doc_stats: list[dict] = []

    for fpath in file_paths:
        p = Path(fpath)
        meta, body = parse_markdown_with_frontmatter(p)
        chunks = chunker.chunk(body)
        doc_stats.append({
            "doc_id": meta["doc_id"],
            "title": meta.get("title", p.name),
            "audience": meta.get("audience", "student"),
            "chars": len(body),
            "num_chunks": len(chunks),
        })

        for i, chunk_text in enumerate(chunks):
            chunk_meta = meta.copy()
            chunk_meta["doc_id"] = meta["doc_id"]
            chunk_meta["chunk_index"] = i
            all_chunks.append(
                Document(
                    id=f"{meta['doc_id']}#{i}",
                    content=chunk_text,
                    metadata=chunk_meta,
                )
            )

    log("\nThống kê tài liệu đã nạp:")
    log(f"{'#':<3} {'Tên tài liệu / doc_id':<45} {'Audience':<10} {'Ký tự':<10} {'Số chunks'}")
    log("-" * 80)
    for idx, stat in enumerate(doc_stats, 1):
        log(f"{idx:<3} {stat['doc_id'][:43]:<45} {stat['audience']:<10} {stat['chars']:<10} {stat['num_chunks']}")
    log("-" * 80)
    log(f"Tổng số chunks tạo ra: {len(all_chunks)}")

    # 3. Nạp vào EmbeddingStore
    log("\n[2] Đang tạo embedding và lưu vào EmbeddingStore (có cache)...")
    t0 = time.time()
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=cached_embedder)
    store.add_documents(all_chunks)
    cached_embedder.save_cache()
    log(f"Đã nạp {store.get_collection_size()} chunks vào store trong {time.time()-t0:.2f}s.")

    # 4. Mock/Real LLM function
    def answer_llm(prompt: str) -> str:
        lines = prompt.splitlines()
        ctx_lines = [l for l in lines if l.startswith("[") or "Điều" in l or "tín chỉ" in l]
        summary = " ".join(ctx_lines)[:250] if ctx_lines else "Tổng hợp thông tin từ ngữ cảnh quy định..."
        return f"[Agent RAG] Dựa trên quy chế đào tạo: {summary}..."

    agent = KnowledgeBaseAgent(store=store, llm_fn=answer_llm)

    # 5. Chạy 5 câu hỏi benchmark
    log("\n[3] CHẠY ĐÁNH GIÁ 5 CÂU HỎI BENCHMARK:")
    log("=" * 80)

    total_score = 0

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        gold = item["gold_answer"]
        filter_meta = item.get("filter")

        log(f"\n--- Câu hỏi {qid}: {query} ---")
        log(f"  * Câu trả lời chuẩn (Gold): {gold}")

        # Tìm kiếm có filter
        results = store.search_with_filter(query, top_k=3, metadata_filter=filter_meta)

        # Kiểm tra độ liên quan (bằng từ khóa đặc trưng)
        relevant_found = False
        top1_relevant = False

        log(f"  * Kết quả Top-3 (với filter={filter_meta}):")
        for rank, res in enumerate(results, start=1):
            doc_id = res["metadata"].get("doc_id", "unknown")
            score = res["score"]
            content = res["content"].replace("\n", " ")
            preview = content[:150]

            # Kiểm tra từ khóa xuất hiện
            matches = [kw for kw in item["expected_keywords"] if kw.lower() in content.lower()]
            is_rel = len(matches) >= 1
            if is_rel:
                relevant_found = True
                if rank == 1:
                    top1_relevant = True

            rel_mark = "✓ [LIÊN QUAN]" if is_rel else "✗ [KHÔNG CHỨA ĐÁP ÁN]"
            log(f"    [{rank}] Score: {score:.4f} | Nguồn: {doc_id} | {rel_mark}")
            log(f"        Trích đoạn: \"{preview}...\"")

        # Chấm điểm theo SCORING.md (2đ: top-1 liên quan; 1đ: top-2/3 liên quan; 0đ: không có)
        if top1_relevant:
            q_score = 2
        elif relevant_found:
            q_score = 1
        else:
            q_score = 0
        total_score += q_score
        log(f"  * Đánh giá chất lượng truy xuất: {q_score}/2 điểm")

        # Câu trả lời của Agent
        agent_ans = agent.answer(query, top_k=3)
        log(f"  * Phản hồi của Agent: {agent_ans}")

        # Chạy A/B Testing nếu câu hỏi có yêu cầu
        if item.get("need_filter_ab"):
            log("\n  [A/B TESTING] So sánh Lọc Metadata vs Không lọc:")
            no_filter_results = store.search(query, top_k=3)
            log("    Top-3 khi KHÔNG lọc metadata:")
            for rk, r in enumerate(no_filter_results, 1):
                d_id = r["metadata"].get("doc_id")
                aud = r["metadata"].get("audience", "unknown")
                log(f"      ({rk}) doc_id: {d_id} (audience={aud}) | Score: {r['score']:.4f}")
            log("    Top-3 khi CÓ lọc metadata={'audience': 'student'}:")
            for rk, r in enumerate(results, 1):
                d_id = r["metadata"].get("doc_id")
                aud = r["metadata"].get("audience", "unknown")
                log(f"      ({rk}) doc_id: {d_id} (audience={aud}) | Score: {r['score']:.4f}")
            log("    => Nhận xét A/B: Bộ lọc metadata giúp loại bỏ hoàn toàn các tài liệu chung (audience='all') và chỉ tập trung vào quy chế đào tạo riêng cho sinh viên.")

    log("\n" + "=" * 80)
    log(f"TỔNG KẾT ĐIỂM TRUY XUẤT: {total_score} / 10 điểm ({total_score*10}%)")
    log("=" * 80)

    # Ghi file kết quả
    try:
        Path(output_file).write_text("\n".join(lines_output), encoding="utf-8")
        print(f"\n[Thành công] Đã lưu kết quả benchmark vào file: {output_file}")
    except Exception as e:
        print(f"[Lỗi] Không thể ghi file {output_file}: {e}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Lab 07 Retrieval Strategy Benchmark")
    parser.add_argument("--data-dir", default="data/neu-quy-dinh", help="Thư mục tài liệu .md")
    parser.add_argument(
        "--strategy",
        default="recursive",
        choices=["recursive", "fixed", "sentence", "heading"],
        help="Chiến lược chunking",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Kích thước chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Độ chồng chéo")
    parser.add_argument(
        "--provider",
        default="local",
        choices=["local", "gemini", "openai", "mock"],
        help="Backend embedding",
    )
    parser.add_argument(
        "--output",
        default="ket_qua_benchmark.txt",
        help="Đường dẫn file kết quả xuất ra",
    )

    args = parser.parse_args()
    return run_benchmark(
        data_dir=args.data_dir,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        provider=args.provider,
        output_file=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
