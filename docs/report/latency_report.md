# Báo cáo đánh giá latency hệ thống

## 1. Mục tiêu

Báo cáo này ghi lại cách đo và kết quả latency hiện tại của hệ thống truy xuất video. Latency được hiểu là thời gian từ lúc client gửi request đến API `/search` cho tới khi server trả về JSON kết quả.

Phần đánh giá này chỉ đo tốc độ truy vấn sau khi hệ thống đã ingest embedding vào Qdrant. Không tính thời gian ingest, thời gian khởi động Docker, thời gian load app, hoặc thời gian người dùng xem video trên giao diện.

## 2. Thiết lập đo

Script đo latency:

```powershell
uv run python eval/scripts/measure_latency.py `
  --queries eval/datasets/preliminary_latency_sample/queries.jsonl `
  --models SigLIP `
  --runs 5 `
  --warmup-runs 1 `
  --summary-out eval/runs/sample_latency_summary.json `
  --per-query-out eval/runs/sample_latency_per_query.jsonl `
  --samples-out eval/runs/sample_latency_samples.jsonl
```

Điều kiện chạy:

- API server: `http://localhost:8000`
- Vector database: Qdrant
- Model truy vấn: `SigLIP`
- Số query mẫu: 9
- Số lần đo mỗi query: 5
- Số lần warmup mỗi query: 1
- Tổng số request được tính metric: 45

Bộ query latency nằm ở:

```text
eval/datasets/preliminary_latency_sample/queries.jsonl
```

## 3. Các metric latency

### mean_ms

`mean_ms` là latency trung bình của tất cả request:

```text
mean_ms = tổng latency / số request
```

Metric này dễ hiểu nhưng có thể bị ảnh hưởng bởi một vài request rất chậm. Vì vậy không nên chỉ nhìn `mean_ms`.

### min_ms và max_ms

`min_ms` là request nhanh nhất, `max_ms` là request chậm nhất trong tập đo.

Hai metric này giúp kiểm tra biên độ dao động, nhưng `max_ms` có thể bị ảnh hưởng bởi outlier.

### p50_ms

`p50_ms` là median latency. Nghĩa là 50% request nhanh hơn hoặc bằng giá trị này.

Đây là metric đại diện cho trải nghiệm thông thường của người dùng.

### p90_ms

`p90_ms` nghĩa là 90% request nhanh hơn hoặc bằng giá trị này.

Metric này cho thấy độ trễ ở nhóm request chậm hơn mức thông thường.

### p95_ms

`p95_ms` nghĩa là 95% request nhanh hơn hoặc bằng giá trị này.

Đây là metric quan trọng để đánh giá trải nghiệm thực tế, vì hệ thống tương tác không chỉ cần nhanh trung bình mà còn cần tránh nhiều request bị chậm.

### p99_ms

`p99_ms` nghĩa là 99% request nhanh hơn hoặc bằng giá trị này.

Metric này dùng để quan sát phần đuôi rất chậm của latency. Với số lượng mẫu nhỏ, `p99_ms` chỉ nên dùng như chỉ báo tham khảo.

### success_count và error_count

`success_count` là số request thành công. `error_count` là số request lỗi hoặc timeout.

Nếu `error_count > 0`, kết quả latency cần được đọc cẩn thận vì hệ thống có thể đang không ổn định.

### mean_result_count

`mean_result_count` là số kết quả trung bình được trả về bởi API `/search`.

Metric này không phải latency metric trực tiếp, nhưng giúp đọc kết quả: nếu một task trả rất ít hoặc 0 kết quả thì latency có thể không phản ánh đầy đủ chi phí xử lý trong tình huống truy vấn thành công.

## 4. Kết quả tổng quan

Kết quả tổng hợp:

| Metric | Giá trị |
|---|---:|
| Request được đo | 45 |
| Success | 45 |
| Error | 0 |
| mean_ms | 2524.005 |
| min_ms | 2225.518 |
| p50_ms | 2297.000 |
| p90_ms | 3047.514 |
| p95_ms | 3069.054 |
| p99_ms | 3161.830 |
| max_ms | 3199.714 |
| mean_result_count | 36.667 |

Nhận xét:

- Hệ thống không có request lỗi trong lần đo này.
- Latency thông thường nằm quanh 2.3 giây.
- 95% request hoàn thành trong khoảng 3.07 giây.
- Request chậm nhất khoảng 3.20 giây.

## 5. Kết quả theo task

| Task | mean_ms | p50_ms | p95_ms | p99_ms | max_ms | mean_result_count |
|---|---:|---:|---:|---:|---:|---:|
| KIS | 2280.803 | 2264.569 | 2346.844 | 2421.514 | 2440.182 | 42.000 |
| Q&A | 2283.994 | 2279.577 | 2317.474 | 2330.141 | 2333.308 | 34.667 |
| TRAKE | 3007.217 | 3009.151 | 3139.445 | 3187.660 | 3199.714 | 33.333 |

### KIS

KIS có p50 khoảng 2.26 giây và p95 khoảng 2.35 giây. Latency khá ổn định giữa các query.

### Q&A

Q&A có latency gần giống KIS. Điều này hợp lý vì hệ thống hiện tại đang xử lý Q&A chủ yếu như truy xuất bằng text query, chưa có bước sinh hoặc kiểm chứng câu trả lời.

### TRAKE

TRAKE chậm hơn KIS và Q&A. p50 khoảng 3.01 giây và p95 khoảng 3.14 giây.

Nguyên nhân hợp lý là TRAKE dùng nhiều sub-query cho một truy vấn và có thêm bước ghép chuỗi sự kiện theo thời gian.

## 6. Đánh giá hiện trạng

Với hệ thống tương tác:

- KIS/Q&A: khoảng 2.3 giây cho một truy vấn đơn.
- TRAKE: khoảng 3.0 giây cho truy vấn nhiều sự kiện.

Tuy vậy, hệ thống đang ổn định vì không có lỗi request và latency không dao động mạnh. Đây là baseline có thể dùng được cho demo nội bộ, nhưng nên tối ưu thêm nếu muốn trải nghiệm tìm kiếm mượt hơn.

## 7. Kết luận

Hệ thống hiện tại chạy ổn định. KIS và Q&A mất khoảng 2.3 giây ở p50, còn TRAKE mất khoảng 3.0 giây ở p50. 

## 8. Notes
- bóc ra 1 frame để mô tả về nó, rồi đi tìm.d