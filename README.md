# Hệ Thống Truy Vấn Video Đa Phương Thức — AIC 2026

Hệ thống truy vấn video tương tác đa phương thức (**Multimodal Video Retrieval System**) được nghiên cứu và phát triển bởi Team **TK22-LQDBD** phục vụ cuộc thi **AI Challenge TP. Hồ Chí Minh 2026**.

Hệ thống được thiết kế để giải quyết các bài toán cốt lõi của cuộc thi với thời gian phản hồi nhanh:
- **Textual Known-Item Search (KIS):** Định vị chính xác video và keyframe mô tả sự kiện/cảnh quay từ câu truy vấn ngôn ngữ tự nhiên.
- **Visual Question Answering (QA):** Tìm kiếm phân cảnh chứa câu trả lời và trích xuất thông tin trực quan (màu sắc, số lượng, chữ viết, hành động).
- **Temporal Action/Event Retrieval (TRAKE):** Truy vấn chuỗi hành động diễn ra tuần tự theo thời gian ($E_1 \rightarrow E_2 \rightarrow \dots \rightarrow E_k$) trong cùng một video.

---

## 🛠️ Quy Trình Xử Lý Dữ Liệu Video (Preprocessing Pipeline)

Dữ liệu video được xử lý offline qua 6 bước để chuẩn bị nguồn dữ liệu cho việc truy vấn:

| STT | Nguồn Dữ Liệu | Thuật Toán & Mô Hình Sử Dụng | Script / Notebook | Đầu Ra |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Trích xuất Keyframe & Tạo Embedding** | • **Keyframe:** Thuật toán so khớp độ tương đồng (similarity score) giữa các frame liên tiếp (không phụ thuộc vào shot detection).<br>• **Vision Embedding:** **SigLIP2** (`google/siglip2-giant-opt-patch16-384`, 1536d) và **Qwen3-VL-Embedding-2B** (2048d) hỗ trợ tiếng Việt trực tiếp. | `notebooks/aic2026-qwen3-vl-embedding.ipynb`<br>`scripts/ingest_named_vectors_incremental.py` | Keyframe ảnh (`.webp`) và vector ngữ nghĩa của từng keyframe. |
| **2** | **Shot Detection** | **AutoShot** (Mô hình phân đoạn ranh giới cảnh quay). | `notebooks/autoshot-aic.ipynb` | Các phân cảnh quay trong video `[t_start, t_end]`. |
| **3** | **Scene Text (OCR)** | **PP-OCRv6** kết hợp **PaddleOCR-VL-1.6**. | `data_processor/ocr/__init__.py`<br>`scripts/ingest_ocr_transcript_incremental.py` | Văn bản và vị trí (bounding box) xuất hiện trong khung hình. |
| **4** | **Audio Transcript** | **OpenAI Whisper** (Mô hình nhận dạng giọng nói tự động). | `notebooks/aic2026-whisper.ipynb`<br>`scripts/ingest_ocr_transcript_incremental.py` | Lời thoại kèm mốc thời gian `[start_time, end_time]`. |
| **5** | **Object Detection** | **YOLOE-26L** (Mô hình Open-Vocabulary Object Detection). | `notebooks/aic2026-yoloe-26l.ipynb` | Danh sách nhãn và số lượng đối tượng trong từng khung hình. |
| **6** | **Shot Captioning** | **Gemini 2.5 Flash Lite** (via Vertex AI) mô tả cảnh + **Qwen3-Embedding-0.6B** (1024d) tạo vector mô tả. | `data_processor/caption`<br>`notebooks/aic2026-qwen3-embedding.ipynb` | Đoạn văn mô tả chi tiết và vector ngữ nghĩa của phân cảnh. |

---

## 🔍 Cơ Chế Truy Vấn Cho Mỗi Nguồn Dữ Liệu (Retrieval Mechanisms)

### 1. Text-Keyframe Similarity
- Câu truy vấn được mã hóa bằng Text Encoder (**SigLIP2** hoặc **Qwen3-VL-Embedding-2B**, cả hai đều hỗ trợ tốt tiếng Việt).
- Hệ thống gửi vector truy vấn tới **Qdrant** để tìm kiếm láng giềng gần nhất (Cosine Similarity) với các vector keyframe đã lưu.
- Điểm số được chuẩn hóa và có thể kết hợp (fusion) điểm giữa các mô hình.

### 2. Text-Shot Caption Similarity
- Câu truy vấn được tìm kiếm đối chiếu với mô tả của các phân cảnh:
  - **Dense Semantic Search:** Mã hóa truy vấn bằng **Qwen3-Embedding** và tìm kiếm vector trên các khía cạnh mô tả shot trong Qdrant.
  - **Lexical BM25 Search:** Tìm kiếm từ khóa chính xác trên toàn bộ văn bản caption lưu tại Elasticsearch.
- Điểm số tương đồng của caption được kết hợp cùng điểm keyframe để tăng độ chính xác cho các câu truy vấn miêu tả hành động chi tiết.

### 3. OCR / Transcript / Object Detection Filter
Các bộ lọc điều kiện được áp dụng đồng thời (AND logic) để lọc bớt các kết quả không phù hợp:
- **Bộ lọc OCR:** Tìm kiếm chuỗi ký tự trên Elasticsearch trong khoảng thời gian quanh keyframe ($\pm 5$ giây) hoặc toàn video.
- **Bộ lọc Transcript:** Tìm kiếm lời thoại qua kết quả Whisper trong khoảng thời gian quanh keyframe ($\pm 2$ giây).
- **Bộ lọc Object Detection:** Lọc trực tiếp trên payload của Qdrant theo nhãn đối tượng và số lượng đối tượng tối thiểu/tối đa trong khung hình (ví dụ: `person >= 2`).

### 4. Thuật toán Temporal Search (Cho chuỗi sự kiện TRAKE)
- Áp dụng khi tìm kiếm chuỗi hành động diễn ra tuần tự ($E_1 \rightarrow E_2 \rightarrow \dots \rightarrow E_k$):
  1. Tìm kiếm độc lập từng sự kiện $E_i$ để lấy danh sách ứng viên keyframe.
  2. Dùng thuật toán **Depth-First Search (DFS)** duyệt theo từng video để tìm chuỗi frame thỏa mãn:
     - Thời gian tăng dần nghiêm ngặt: $t_1 < t_2 < \dots < t_k$.
     - Khoảng cách giữa 2 sự kiện liên tiếp không vượt quá ngưỡng: $0 < \Delta \text{frame} \le \texttt{MAX\_FRAME\_GAP}$ (mặc định 2000 frame).
  3. Xếp hạng chuỗi ứng viên theo điểm tương đồng trung bình của các sự kiện.
  4. Hỗ trợ gom nhóm theo shot (Group Shots) hoặc theo video (Group Video) để tránh trùng lặp khung hình.

---

## 💻 Công Nghệ Sử Dụng (Tech Stack)

| Thành Phần | Công Nghệ | Vai Trò & Chức Năng |
| :--- | :--- | :--- |
| **Vector Database** | **Qdrant** | • Lưu vision embedding của keyframe (`SigLIP2`, `Qwen3_VL_Embedding`).<br>• Payload lưu nhãn và số lượng đối tượng từ Object Detection.<br>• Lưu shot caption embedding (có quantize int4 để giảm dung lượng bộ nhớ). |
| **Text Search Engine** | **Elasticsearch 8.15** | • Lưu dữ liệu văn bản OCR và audio transcript.<br>• Tìm kiếm toàn văn bản (full-text search) bằng thuật toán BM25 và fuzzy match. |
| **Backend API** | **FastAPI + Uvicorn** | • Xây dựng RESTful API cho ứng dụng web và điều phối các luồng truy vấn.<br>• Tích hợp DRES Client để đăng nhập và nộp bài trực tiếp lên hệ thống thi. |
| **Containerization** | **Docker & Docker Compose** | • Đóng gói và chạy toàn bộ ứng dụng (`app`, `qdrant`, `elasticsearch`) bằng một lệnh.<br>• Hỗ trợ chạy linh hoạt trên cả CPU và GPU NVIDIA. |

---

## 🚀 Hướng Dẫn Chạy Ứng Dụng Bằng Docker

### 1. Chuẩn bị File Cấu Hình

Tạo file cấu hình `.env` từ file mẫu:
```bash
cp .env.example .env
```

Chỉnh sửa thông tin tài khoản DRES trong file `.env`:
```env
DRES_BASE_URL=https://eventretrieval.one
DRES_DEFAULT_USERNAME=team_197
DRES_DEFAULT_PASSWORD=mat_khau_cua_doi
DRES_COOKIE_SECURE=false
```

### 2. Khởi Chạy Ứng Dụng

#### Chạy trên CPU (Mặc định)
```bash
docker compose up -d --build
```

#### Chạy với GPU NVIDIA (Tăng tốc mô hình)
Yêu cầu máy chủ đã cài [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

### 3. Kiểm Tra Trạng Thái & Truy Cập

Kiểm tra trạng thái các container:
```bash
docker compose ps
```

Xem log ứng dụng:
```bash
docker compose logs -f app
```

Mở trình duyệt truy cập giao diện tại:
👉 **`http://localhost:8000`**

### 4. Chạy Lệnh Tiện Ích Trong Docker (Nếu cần)

```bash
# Trích xuất metadata FPS video (keyframe -> timestamp)
docker compose run --rm app python utils/video_metadata.py

# Ingest dữ liệu vào Qdrant & Elasticsearch
docker compose run --rm app python -c "from retrieval_system import RetrievalSystem; RetrievalSystem(re_ingest=True)"

# Chạy truy vấn qua CLI
docker compose run --rm app python skills/aic-video-retrieval/scripts/query_runner.py --query "người đi xe máy" --limit 10
```

### 5. Dừng Ứng Dụng
```bash
# Dừng container
docker compose down
```
