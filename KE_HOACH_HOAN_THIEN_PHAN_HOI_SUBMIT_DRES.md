# Kế hoạch hoàn thiện phản hồi nộp bài DRES

Trạng thái: **Đã triển khai — chờ kiểm thử trực quan và smoke test DRES chính thức**

Nguồn đối chiếu:

- `SubmitSystem/Client-Examples/oas-client.json`
- Client mẫu Java, Kotlin và Angular trong `SubmitSystem/Client-Examples/`
- Đặc tả Chung kết AIC 2026 trong `HD-ChungKet-2026.pdf`

## 1. Mục tiêu

Sau mỗi lần nhấn nộp KIS, Q&A hoặc TRAKE, người dùng phải biết rõ:

1. Request có gửi được tới DRES hay không.
2. DRES có tiếp nhận gói tin hay không.
3. Kết quả đang chờ chấm hay đã có verdict.
4. Verdict là đúng, sai, chưa xác định hay không thể chấm.
5. DRES trả về mô tả gì.
6. Request đã gửi tới evaluation và URL nào, nhưng không làm lộ session.

Không được dùng một thông báo chung như `accepted` cho tất cả trường hợp.

## 2. Phản hồi chuẩn cần hỗ trợ

Endpoint submit chính thức:

```text
POST /api/v2/submit/{evaluationId}?session=<SESSION_ID>
```

Body thành công theo OpenAPI:

```json
{
  "status": true,
  "submission": "CORRECT",
  "description": "Submission accepted"
}
```

Các verdict hợp lệ:

| Verdict | Ý nghĩa trên giao diện | Màu đề xuất |
| --- | --- | --- |
| `CORRECT` | Đáp án đúng | Xanh lá |
| `WRONG` | Đáp án sai; gói tin vẫn đã được DRES nhận | Đỏ |
| `INDETERMINATE` | Chưa xác định được kết quả | Vàng |
| `UNDECIDABLE` | Không thể chấm tự động | Vàng |

Quy tắc theo HTTP status:

| HTTP | Ý nghĩa | Hành vi giao diện |
| --- | --- | --- |
| `200` | DRES đã nhận và có verdict | Hiển thị verdict, description và trạng thái tiếp nhận |
| `202` | DRES đã nhận nhưng chưa có verdict | Hiển thị `Đã nhận — đang chờ kết quả` |
| `400` | Request/payload không hợp lệ | Hiển thị lý do từ `description` |
| `401` | Session hết hạn hoặc không hợp lệ | Xóa phiên cục bộ và yêu cầu đăng nhập lại |
| `404` | Sai evaluation hoặc không có task đang nhận bài | Không báo thành công |
| `412` | DRES từ chối submission | Hiển thị `Bị DRES từ chối` cùng lý do |

## 3. Thiết kế phản hồi nội bộ của backend

Backend chuẩn hóa phản hồi DRES trước khi trả cho frontend:

```json
{
  "accepted": true,
  "pending": false,
  "dresHttpStatus": 200,
  "verdict": "CORRECT",
  "description": "Submission accepted",
  "evaluationId": "evaluation-id",
  "destination": "https://eventretrieval.one/api/v2/submit/evaluation-id?session=<SESSION_ID_ẨN>",
  "payload": {
    "answerSets": []
  },
  "rawResult": {
    "status": true,
    "submission": "CORRECT",
    "description": "Submission accepted"
  }
}
```

Quy tắc backend:

- [x] Chỉ đặt `accepted=true` khi HTTP là `200`/`202` **và** body có `status === true`.
- [x] Với HTTP `200`, kiểm tra `submission` thuộc enum verdict hợp lệ.
- [x] Với HTTP `202`, đặt `pending=true`; không tự suy đoán đúng/sai.
- [x] Nếu HTTP `200` nhưng thiếu `status`, `status=false` hoặc body sai schema, trả lỗi rõ ràng; không ghi nhận là đã nộp thành công.
- [x] Giữ nguyên `description` của DRES để người dùng biết lý do.
- [x] Không trả DRES session thật về frontend, log hoặc modal.
- [x] Chỉ ghi nhận submission thành công sau khi phản hồi đã vượt qua bước kiểm tra trên.

## 4. Thay đổi backend dự kiến

### 4.1. `services/dres_client.py`

- [x] Thêm validator cho `SuccessfulSubmissionsStatus`.
- [x] Tách rõ HTTP status và JSON body.
- [x] Kiểm tra body là object và có `status` kiểu boolean.
- [x] Chuẩn hóa lỗi `400`, `401`, `404`, `412` từ `ErrorStatus.description`.

### 4.2. `app.py`

- [x] Thay `accepted: True` cố định bằng kết quả đã xác thực.
- [x] Trả trực tiếp các trường `verdict`, `description`, `dresHttpStatus`, `pending` và URL đã che session.
- [x] Với `401`, xóa session cục bộ như hiện tại.
- [x] Với phản hồi thành công sai schema, trả `502` và thông báo rõ ràng.
- [x] Không đánh dấu fingerprint/duplicate trước khi DRES xác nhận tiếp nhận.

## 5. Sửa cơ chế chống nộp trùng

OpenAPI của `currentTask` không cung cấp task ID duy nhất; nó chỉ có `name`,
`taskGroup`, `taskType` và `duration`. Vì vậy fingerprint lưu suốt session có thể
chặn nhầm một submission hợp lệ ở task sau.

Kế hoạch:

- [x] Giữ khóa nút khi request đang chạy để chống double-click.
- [x] Thay duplicate vĩnh viễn bằng cửa sổ chống lặp 3 giây.
- [x] Sau cửa sổ chống lặp, cho phép người dùng chủ động nộp lại.
- [x] Không dùng `name/taskType/taskGroup` như định danh duy nhất của một lượt task.
- [x] Giữ bước xác nhận trước mỗi lần nộp lại.

## 6. Thiết kế giao diện kết quả

### 6.1. KIS và Q&A

Trong modal hiện tại, sau khi submit hiển thị một thẻ kết quả gồm:

```text
Gói tin đã được DRES tiếp nhận
HTTP: 200
Verdict: ĐÚNG (CORRECT)
Mô tả: Submission accepted
Evaluation: <evaluationId>
```

Nếu verdict là `WRONG`:

```text
Gói tin đã được DRES tiếp nhận
Verdict: SAI (WRONG)
```

Không dùng câu `Nộp bài thất bại` cho verdict `WRONG`, vì request đã được nhận;
chỉ có đáp án là sai.

### 6.2. TRAKE

- [x] Hiển thị cùng cấu trúc verdict như KIS/Q&A trong workspace TRAKE.
- [x] Không tự xóa chuỗi TRAKE sau verdict `WRONG`, để người dùng có thể kiểm tra và sửa.
- [x] Chỉ cung cấp nút `Xóa chuỗi` hoặc `Nộp lại`; không tự động nộp lần hai.

### 6.3. Chi tiết kỹ thuật

- [x] Hiển thị URL đích đã che session như hiện tại.
- [x] Thêm vùng `Phản hồi DRES` chứa JSON đã được làm sạch.
- [x] Có nút sao chép payload và phản hồi để hỗ trợ kiểm tra nhanh.
- [x] Màu sắc không phải tín hiệu duy nhất; luôn có chữ `ĐÚNG`, `SAI`, `ĐANG CHỜ`, `BỊ TỪ CHỐI`.

## 7. Nâng cấp Mock DRES

Mock phải trả đúng schema của DRES thật:

```json
{
  "status": true,
  "submission": "CORRECT",
  "description": "Mock submission accepted"
}
```

Các thay đổi:

- [x] Sửa response submit theo `SuccessfulSubmissionsStatus`.
- [x] Sửa logout thành `{ "status": true, "description": "..." }`.
- [x] Thêm lựa chọn `Phản hồi cho lần nộp kế tiếp` tại trang `/debug`:
  - `200 CORRECT`
  - `200 WRONG`
  - `200 INDETERMINATE`
  - `200 UNDECIDABLE`
  - `202 PENDING`
  - `412 REJECTED`
  - `401 SESSION EXPIRED`
- [x] Sau một lần submit, tùy chọn phản hồi trở về mặc định `200 CORRECT` để tránh quên trạng thái test.
- [x] Debug vẫn ghi method, path, query đã che session, evaluation ID và payload.

## 8. Kiểm thử tự động

### 8.1. DRES client

- [x] Login `200` có `sessionId`.
- [x] Login `200` thiếu `sessionId` bị xem là phản hồi lỗi.
- [x] Submit `200 + status=true + CORRECT`.
- [x] Submit `200 + status=true + WRONG`.
- [x] Submit `200 + status=false` không được báo thành công.
- [x] Submit `200` thiếu trường bắt buộc không được báo thành công.
- [x] Submit `202` được đánh dấu pending.
- [x] Các lỗi `400`, `401`, `404`, `412` giữ đúng HTTP status và description.

### 8.2. Payload

- [x] KIS giữ `mediaItemName`, `start`, `end` theo millisecond.
- [x] Q&A giữ định dạng `QA-ANSWER-VIDEO_ID-TIME_MS`.
- [x] TRAKE giữ định dạng `TR-VIDEO_ID-FRAME_IDS` và thứ tự tăng nghiêm ngặt.

### 8.3. Backend và session

- [x] Không trả session DRES thật cho trình duyệt.
- [x] Không ghi fingerprint khi DRES từ chối hoặc body không hợp lệ.
- [x] Cho phép nộp lại có chủ đích sau cửa sổ chống lặp.
- [x] Double-click trong lúc request chạy chỉ tạo một request.

### 8.4. Mock end-to-end

- [x] Một evaluation `mock-final` dùng cho cả KIS, Q&A và TRAKE.
- [ ] UI hiển thị đúng cho toàn bộ verdict và trạng thái lỗi mô phỏng.
- [ ] Gói tin trong `/debug` trùng với payload xem trước trên giao diện.

## 9. Kiểm thử thủ công trước khi dùng thật

1. Chạy Mock DRES và ứng dụng ở chế độ mock.
2. Lần lượt cấu hình và nộp thử tất cả trạng thái ở mục 7.
3. Kiểm tra thông báo giao diện không nhầm `WRONG` với lỗi truyền request.
4. Kiểm tra `202` không bị hiển thị thành đúng hoặc sai.
5. Kiểm tra `412` không bị ghi nhận là đã nộp thành công.
6. Kiểm tra session không xuất hiện trong HTML, JavaScript hoặc JSON trả về frontend.
7. Sau khi toàn bộ test mock đạt, chạy smoke test login/list evaluation/current task với DRES thật.
8. Chỉ gửi một submission thật khi có task/evaluation thử nghiệm được ban tổ chức cho phép.

## 10. Tiêu chí nghiệm thu

- [x] Login, evaluation, current task và logout đúng OpenAPI.
- [x] Backend không còn `accepted: True` cố định.
- [x] Mock DRES trả đúng schema chính thức.
- [x] Giao diện có trạng thái riêng cho: đã nhận, đang chờ, đúng, sai và bị từ chối.
- [x] Verdict `WRONG` không bị mô tả là lỗi gửi request.
- [x] `description` từ DRES được hiển thị nguyên nghĩa.
- [x] Không lộ password hoặc session.
- [x] Không chặn nhầm submission ở task sau bằng fingerprint vĩnh viễn.
- [x] Tất cả test tự động đạt (`24 tests`).
- [x] README được cập nhật với ví dụ phản hồi thật.

## 11. Thứ tự triển khai đề xuất

1. Viết validator và test phản hồi DRES.
2. Chuẩn hóa response của backend `/api/dres/submit`.
3. Sửa Mock DRES theo OpenAPI.
4. Sửa UI KIS/Q&A.
5. Sửa UI TRAKE.
6. Thay cơ chế chống duplicate.
7. Chạy test tự động và toàn bộ kịch bản Mock.
8. Cập nhật README.
9. Review thay đổi và chỉ sau khi được phê duyệt mới push lên `main`.

## 12. Phê duyệt

- [x] Đồng ý triển khai toàn bộ kế hoạch.
- [ ] Cần điều chỉnh kế hoạch trước khi triển khai.
