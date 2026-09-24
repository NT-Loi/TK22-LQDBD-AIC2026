# Hướng dẫn kiểm thử giao diện bằng Mock DRES

Mock DRES cho phép thử toàn bộ luồng đăng nhập, chọn evaluation và nộp KIS/Q&A/TRAKE ngay trên giao diện mà không cần DRES chính thức hoạt động. Tất cả request chỉ đi tới `127.0.0.1`; không có submission nào được chuyển tiếp ra Internet.

## 1. Khởi động

Nếu ứng dụng thật đang chạy ở cổng `8000`, dừng nó bằng `Ctrl+C` trước để tránh trùng cổng.

Mở terminal thứ nhất:

```bash
uv run uvicorn scripts.mock_dres_server:app --host 127.0.0.1 --port 19100
```

Mở terminal thứ hai:

```bash
uv run --env-file .env.mock.example uvicorn app:app --host 127.0.0.1 --port 8765 --reload
```

Mở hai trang:

- Giao diện ứng dụng: <http://127.0.0.1:8765>
- Gói tin Mock DRES nhận được: <http://127.0.0.1:19100/debug>

Nếu trình duyệt từng mở phiên bản cũ, nhấn `Ctrl+Shift+R` một lần.

## 2. Đăng nhập mock

1. Nhấn **Đăng nhập DRES**.
2. Nhấn **Đăng nhập mặc định (team_197)**.
3. Chọn evaluation duy nhất `MOCK • Chung kết AIC 2026`.
4. Dùng dropdown ở đầu sidebar để chọn `KIS`, `Q&A` hoặc `TRAKE`.
5. Header sẽ hiển thị tài khoản và evaluation đang chọn; task mock có loại `MANUAL` nên không tự ghi đè lựa chọn của bạn.

Password trong `.env.mock.example` chỉ là `mock-password`, không phải credential thật.

## 3. Kiểm thử KIS

1. Chọn `KIS` trong dropdown **Loại submission**.
2. Tìm một kết quả rồi nhấn **Submit**, hoặc mở video và dừng đúng khoảnh khắc mong muốn.
3. Khi nhấn Submit trong video, video phải dừng ngay tại vị trí hiện tại.
4. Kiểm tra modal xác nhận có đúng video, frame và millisecond.
5. Payload phải có dạng:

```json
{
  "answerSets": [{
    "answers": [{
      "mediaItemName": "L24_V044",
      "start": 7520,
      "end": 7520
    }]
  }]
}
```

6. Nhấn **Xác nhận nộp KIS**. Kết quả mặc định phải hiển thị
   `DRES ĐÃ NHẬN • ĐÚNG`, HTTP `200`, verdict `CORRECT` và mô tả từ Mock DRES.
7. Mở trang debug và đối chiếu `path`, `evaluationId` cùng `payload`.

## 4. Kiểm thử Q&A

1. Chọn `Q&A` trong dropdown **Loại submission**.
3. Chọn kết quả hoặc dừng video rồi nhấn Submit.
4. Nhập đáp án tiếng Việt, ví dụ `màu đỏ`.
5. Payload preview phải có dạng:

```text
QA-màu đỏ-L24_V044-7520
```

6. Không được phép nộp khi đáp án trống.
7. Xác nhận và kiểm tra gói tin trên trang debug.

## 5. Kiểm thử TRAKE

1. Chọn `TRAKE` trong dropdown **Loại submission**.
2. Nút Submit trên kết quả phải đổi thành **Thêm vào TRAKE**; nút trong video đổi thành **Thêm mốc TRAKE**.
3. Dừng video ở nhiều vị trí và thêm các frame theo thứ tự tăng dần.
4. Kiểm tra TRAKE Workspace cho phép mở, thay, di chuyển và xóa mốc.
5. Nếu thứ tự frame không tăng nghiêm ngặt, nút **Nộp TRAKE** phải bị khóa.
6. Chuỗi preview hợp lệ có dạng:

```text
TR-L24_V044-12,48,103
```

7. Nhấn **Nộp TRAKE** và kiểm tra gói tin trên trang debug.

## 6. Kiểm thử verdict và lỗi

Tại `http://127.0.0.1:19100/debug`, chọn **Phản hồi cho lần nộp kế tiếp** rồi
nhấn **Áp dụng** trước khi nộp trên giao diện ứng dụng:

| Lựa chọn | Kết quả mong đợi trên ứng dụng |
| --- | --- |
| `CORRECT` | `DRES ĐÃ NHẬN • ĐÚNG` |
| `WRONG` | `DRES ĐÃ NHẬN • SAI`; không được ghi là lỗi gửi request |
| `INDETERMINATE` | `DRES ĐÃ NHẬN • CHƯA XÁC ĐỊNH` |
| `UNDECIDABLE` | `DRES ĐÃ NHẬN • KHÔNG THỂ CHẤM` |
| `PENDING` | `DRES ĐÃ NHẬN • ĐANG CHỜ VERDICT`, HTTP DRES `202` |
| `REJECTED` | `DRES TỪ CHỐI SUBMISSION`, HTTP `412` |
| `SESSION_EXPIRED` | Yêu cầu đăng nhập lại, HTTP `401` |

Mỗi lựa chọn chỉ áp dụng cho một lần nộp. Sau đó Mock tự trở về `CORRECT`.
Mở mục **Phản hồi JSON từ DRES** để kiểm tra `status`, `submission` và
`description`. Session trong URL phải luôn là `<SESSION_ID_ẨN>`.

## 7. Những điểm cần đánh giá về độ dễ dùng

- Có nhận ra tài khoản và evaluation hiện tại ngay không?
- Có phân biệt rõ KIS, Q&A và TRAKE không?
- Modal có đủ thông tin để tránh nộp nhầm video/thời điểm không?
- Việc dừng video rồi Submit có tự nhiên không?
- TRAKE Workspace có dễ hiểu thứ tự Event 1, Event 2, ... không?
- Thông báo lỗi có giúp sửa đúng vấn đề không?
- Có thao tác nào cần quá nhiều lần nhấn không?

Bạn có thể ghi nhận xét trực tiếp vào phần cuối của `KE_HOACH_DRES_LOGIN_SUBMIT.md`.

## 8. Kết thúc kiểm thử

Nhấn `Ctrl+C` ở cả hai terminal. Khi chạy lại ứng dụng bình thường mà không có `--env-file .env.mock.example`, backend sẽ quay về `DRES_BASE_URL` trong `.env` hoặc DRES chính thức mặc định.
