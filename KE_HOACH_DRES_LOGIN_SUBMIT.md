# Kế hoạch cập nhật đăng nhập và nộp bài DRES

Trạng thái: Đã hiện thực và kiểm thử bằng Mock DRES; chưa smoke test với DRES chính thức.
Tài liệu yêu cầu: `HD-ChungKet-2026.pdf`
Phạm vi: Cập nhật nút đăng nhập và toàn bộ các nút nộp bài cho KIS, Q&A và TRAKE.

> Kế hoạch đã được phê duyệt và hiện thực. Các ô kiểm tra bên dưới được cập nhật theo kết quả kiểm thử hiện tại.

## 1. Mục tiêu

- Thay phần đăng nhập giả lập bằng kết nối thật tới DRES.
- Cho phép đăng nhập nhanh bằng tài khoản mặc định đã được cung cấp.
- Cho phép người dùng chuyển sang tài khoản DRES khác bất kỳ lúc nào.
- Lấy danh sách kỳ đánh giá đang hoạt động và cho phép chọn `evaluationID`.
- Chuẩn hóa tất cả các nút Submit đang có trong giao diện.
- Với KIS và Q&A, vị trí video đang dừng là vị trí được nộp.
- Với TRAKE, gom nhiều semantic keyframe theo đúng thứ tự sự kiện rồi mới nộp một lần.
- Tạo đúng định dạng dữ liệu trong tài liệu chung kết.
- Hạn chế nộp nhầm, nhấn hai lần và nộp trùng trong cùng một truy vấn.
- Hiển thị đúng phản hồi từ DRES thay vì luôn báo thành công.

## 2. Các quyết định thiết kế chính

### 2.1 Submit thông thường

Hệ thống có nhiều nút Submit ở các vị trí khác nhau. Tất cả sẽ dùng chung một bộ xử lý, nhưng nguồn lấy thời điểm phụ thuộc nơi người dùng nhấn:

- Submit trực tiếp trên thẻ kết quả: dùng `video_id`, `keyframe_index` và `fps` của thẻ.
- Submit trong trình phát video: dùng chính xác vị trí video đang dừng tại lúc nhấn.
- Khi nhấn Submit trong trình phát:
  1. Dừng video ngay lập tức.
  2. Chụp `currentTime` tại thời điểm nhấn.
  3. Tính `timeMs = round(currentTime * 1000)`.
  4. Tính frame hiển thị bằng `round(currentTime * fps)`.
  5. Dùng snapshot này cho màn hình xác nhận và payload DRES.
- Keyframe dùng để mở video chỉ là điểm bắt đầu xem, không được ghi đè vị trí người dùng đã chọn sau khi tua video.

### 2.2 TRAKE là một luồng riêng

TRAKE không dùng hành vi “nhấn Submit và gửi ngay” của KIS/Q&A.

- Trong chế độ TRAKE, nút trên thẻ hoặc trình phát là **Thêm mốc TRAKE**.
- Mỗi lần thêm chỉ đưa một frame vào danh sách tạm, chưa gửi DRES.
- Người dùng kiểm tra, sắp xếp, thay thế hoặc xóa các mốc.
- Tất cả các mốc phải thuộc cùng một video.
- Thứ tự trong danh sách là thứ tự semantic keyframe được gửi.
- Chỉ nút riêng **Nộp TRAKE** mới gửi toàn bộ chuỗi.
- TRAKE gửi frame ID, không gửi millisecond trong chuỗi `TR-...`.

### 2.3 Tài khoản mặc định

- Username mặc định: `team_197`.
- Password mặc định: dùng giá trị đã cung cấp trong bản kế hoạch cũ, lưu tại `DRES_DEFAULT_PASSWORD` trong `.env` cục bộ.
- Không ghi mật khẩu thật vào Markdown, JavaScript, `localStorage`, log hoặc `.env.example`.
- Nút đăng nhập cho hai lựa chọn:
  - **Đăng nhập mặc định**: backend dùng tài khoản mặc định.
  - **Dùng tài khoản khác**: người dùng nhập username/password.
- Sau khi đăng nhập có thể **Đổi tài khoản** hoặc **Đăng xuất**.
- Khi đổi tài khoản, xóa phiên và evaluation của tài khoản cũ trước khi tạo phiên mới.

## 3. Phạm vi không thực hiện

- Không triển khai hoặc thay thế máy chủ DRES.
- Không tự tính điểm chính thức; DRES là nguồn kết quả cuối cùng.
- Không tự xác định đáp án Q&A.
- Không chụp, ghi hình hoặc tải lên đoạn Video KIS.
- Không tự động gửi ngay khi video dừng; người dùng vẫn phải nhấn Submit và xác nhận.

## 4. Hiện trạng cần thay đổi

- `app.py` đang trả về `sessionId` và evaluation giả.
- `/api/submit` chỉ in dữ liệu ra terminal, chưa gọi DRES.
- `static/js/api.js` gửi body đăng nhập rỗng.
- `static/js/main.js` lưu DRES `sessionId` trong `localStorage`.
- `static/js/results.js` và `static/js/video-player.js` có logic Submit tách rời.
- Chưa có ô nhập đáp án Q&A.
- Chưa có vùng làm việc để gom và sắp thứ tự keyframe TRAKE.
- Frontend luôn hiện `Success!` dù chưa nhận verdict thật từ DRES.

## 5. Đối chiếu yêu cầu từ PDF

| Mã | Yêu cầu | Cách đáp ứng | Tiêu chí kiểm tra |
|---|---|---|---|
| YC01 | Đăng nhập bằng username/password | Gọi `POST /api/v2/login` qua backend | Nhận đúng user và `sessionId` |
| YC02 | Lấy `evaluationID` | Gọi `GET /api/v2/client/evaluation/list` | Hiển thị evaluation ACTIVE |
| YC03 | Submit theo evaluation đã chọn | Gọi `POST /api/v2/submit/{evaluationID}` | URL chứa đúng evaluation ID |
| YC04 | KIS dùng video và thời gian | Gửi `mediaItemName`, `start`, `end` | Payload đúng `answerSets` |
| YC05 | Thời gian dùng millisecond | Tính từ vị trí video hoặc frame/fps | `start`, `end` là số nguyên millisecond |
| YC06 | Tên video không có phần mở rộng | Chuẩn hóa basename | Không còn phần mở rộng video |
| YC07 | Q&A dùng chuỗi quy định | Tạo `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>` | Giữ nguyên tiếng Việt của đáp án |
| YC08 | TRAKE dùng chuỗi frame ID | Tạo `TR-<VIDEO_ID>-<FRAME_ID1>,...` | Đúng video và đúng thứ tự frame |
| YC09 | Không nộp trùng cùng truy vấn | Khóa khi gửi và lưu fingerprint sau thành công | Không gửi lặp cùng payload/task |
| YC10 | Textual KIS và Video KIS cùng dạng nộp | Cả hai dùng luồng KIS | Cùng một loại payload |
| YC11 | Nộp sai bị trừ điểm, nộp sớm có lợi | Cảnh báo trước lần gửi cuối | Người dùng xác nhận rõ ràng |
| YC12 | TRAKE có thể đúng toàn phần/một phần | Gửi trọn chuỗi, hiển thị verdict | Không tự tính điểm cục bộ |
| YC13 | Xem truy vấn trên DRES | Có nút mở DRES và làm mới task | Truy cập nhanh DRES |

## 6. Luồng đăng nhập

### 6.1 Chưa đăng nhập

- Nút hiển thị **Đăng nhập DRES**.
- Khi nhấn Submit mà chưa đăng nhập, mở hộp đăng nhập nhưng không làm mất video, vị trí dừng hoặc danh sách TRAKE.

### 6.2 Đăng nhập mặc định

1. Người dùng nhấn **Đăng nhập DRES**.
2. Hộp thoại hiển thị username mặc định `team_197`.
3. Người dùng nhấn **Đăng nhập mặc định**.
4. Frontend chỉ yêu cầu backend dùng tài khoản mặc định, không nhận password.
5. Backend đọc username/password từ `.env`.
6. Backend gọi DRES `/api/v2/login` và giữ `sessionId`.
7. Backend lấy danh sách evaluation.
8. Frontend cho chọn evaluation ACTIVE.

### 6.3 Tài khoản khác

1. Người dùng chọn **Dùng tài khoản khác**.
2. Nhập username/password.
3. Password chỉ tồn tại trong request đăng nhập và không được lưu.
4. Nếu thành công, thực hiện cùng luồng chọn evaluation.
5. Tài khoản tùy chỉnh chỉ có hiệu lực cho phiên hiện tại.

### 6.4 Đã đăng nhập

Ví dụ trạng thái:

```text
team_197 • AIC 2026 Final • Đã kết nối
```

Menu tài khoản gồm:

- Đổi evaluation
- Làm mới task hiện tại
- Mở DRES
- Đổi tài khoản
- Đăng xuất

## 7. Luồng Submit KIS

Áp dụng cho Textual KIS và Video KIS.

### 7.1 Từ thẻ kết quả

1. Lấy `video_id`, `keyframe_index`, `fps` từ kết quả.
2. Tính `timeMs = round(keyframe_index / fps * 1000)`.
3. Mở hộp xác nhận, không gửi ngay.
4. Chỉ gửi sau khi người dùng nhấn **Xác nhận nộp KIS**.

### 7.2 Từ trình phát video

1. Người dùng mở video và phát, tua hoặc di chuyển từng frame.
2. Người dùng dừng tại đúng khoảnh khắc muốn nộp.
3. Khi nhấn Submit, video được pause ngay.
4. `currentTime` là nguồn dữ liệu cuối cùng.
5. Tính `timeMs = round(currentTime * 1000)`.
6. Tính `frameId = round(currentTime * fps)` để hiển thị và kiểm tra.
7. Hộp xác nhận dùng snapshot này, kể cả khi video thay đổi sau đó.

### 7.3 Payload KIS

```json
{
  "answerSets": [{
    "answers": [{
      "mediaItemName": "L21_V001",
      "start": 12400,
      "end": 12400
    }]
  }]
}
```

Quy tắc:

- `mediaItemName` không có phần mở rộng.
- `start` và `end` là số nguyên và bằng nhau cho một khoảnh khắc.
- Vị trí video đang dừng ưu tiên hơn keyframe ban đầu.

## 8. Luồng Submit Q&A

Q&A lấy vị trí giống KIS nhưng cần thêm đáp án văn bản.

1. Người dùng chọn kết quả hoặc dừng video tại khoảnh khắc cần nộp.
2. Nhấn Submit ở chế độ Q&A.
3. Nhập câu trả lời ngắn gọn bằng tiếng Việt.
4. Trim khoảng trắng đầu/cuối nhưng giữ nguyên dấu tiếng Việt.
5. Hiển thị toàn bộ chuỗi sẽ gửi để xác nhận.

```json
{
  "answerSets": [{
    "answers": [{
      "text": "QA-màu đỏ-L21_V001-12400"
    }]
  }]
}
```

Không cho gửi nếu đáp án rỗng.

## 9. Luồng TRAKE riêng

### 9.1 TRAKE Workspace

Thêm panel hoặc drawer **TRAKE Workspace** gồm:

- Video đang chọn.
- Danh sách semantic keyframe theo thứ tự Event 1, Event 2, ...
- Thumbnail, frame ID và thời gian tham khảo.
- Nút phát video tại mốc.
- Nút di chuyển lên/xuống.
- Nút thay thế frame.
- Nút xóa frame hoặc xóa toàn bộ.
- Preview chuỗi `TR-...`.
- Nút riêng **Nộp TRAKE**.

### 9.2 Thêm từ thẻ kết quả

- Dùng nút **Thêm vào TRAKE**, không gửi ngay.
- Frame thêm vào là `keyframe_index` của thẻ.
- Frame đầu tiên xác lập video TRAKE.
- Nếu frame mới thuộc video khác, cho phép hủy hoặc xóa chuỗi cũ để chuyển video; không tự trộn hai video.

### 9.3 Thêm từ trình phát video

- Người dùng dừng video tại semantic keyframe.
- Nhấn **Thêm mốc TRAKE**.
- Video được pause và chụp `currentTime`.
- Tính `frameId = round(currentTime * fps)`.
- Thêm frame vào cuối danh sách; chưa gửi DRES.

### 9.4 Nạp kết quả temporal search

- Nếu kết quả có `temporal_sequence` hoặc `frames`, có nút **Nạp chuỗi vào TRAKE**.
- Nạp theo thứ tự sự kiện, không theo điểm số.
- Người dùng xem lại và tinh chỉnh từng mốc trước khi nộp.

### 9.5 Điều kiện nộp TRAKE

- Đã đăng nhập và chọn evaluation.
- Có ít nhất một frame.
- Tất cả frame thuộc cùng video.
- Không có frame trùng.
- Frame là số nguyên không âm.
- Thứ tự hiển thị là thứ tự muốn gửi.

### 9.6 Payload TRAKE

```json
{
  "answerSets": [{
    "answers": [{
      "text": "TR-L21_V001-310,525,870"
    }]
  }]
}
```

Quy tắc:

- Chỉ một video trong mỗi submission.
- Dùng frame ID, không chuyển thành millisecond.
- Giữ thứ tự sự kiện do người dùng chọn, nhưng chỉ cho phép nộp khi các `frame_id` tăng nghiêm ngặt; nếu việc sắp xếp tay làm sai thứ tự thời gian thì hiển thị cảnh báo và khóa nút nộp.
- Không nộp từng frame riêng lẻ.

## 10. Hộp xác nhận

Hiển thị:

- Tài khoản và evaluation hiện tại.
- Task hiện tại nếu DRES cung cấp.
- Loại bài: KIS, Q&A hoặc TRAKE.
- Video, frame, millisecond hoặc danh sách frame tương ứng.
- Payload cuối cùng.
- Cảnh báo nộp sai bị trừ 10 điểm và thời điểm nộp ảnh hưởng điểm.

Nút:

- **Quay lại chỉnh sửa**.
- **Xác nhận nộp**.
- Khóa nút xác nhận ngay sau lần nhấn đầu tiên.
- Hiển thị **Đang nộp...** trong lúc chờ.

## 11. Thiết kế backend

### 11.1 `services/dres_client.py`

- Dùng `httpx.AsyncClient` với timeout rõ ràng.
- Đăng nhập: `/api/v2/login`.
- Lấy evaluation: `/api/v2/client/evaluation/list`.
- Lấy task: `/api/v2/client/evaluation/currentTask/{evaluationId}`.
- Submit: `/api/v2/submit/{evaluationId}`.
- Đăng xuất: `/api/v2/logout`.
- Chuẩn hóa lỗi JSON/text.
- Không ghi credential hoặc token vào log.

### 11.2 `services/dres_session.py`

- Sinh mã phiên cục bộ ngẫu nhiên.
- Lưu DRES `sessionId` trong bộ nhớ backend.
- Browser chỉ giữ cookie HTTP-only, same-site.
- Frontend không đọc DRES token.
- Xóa phiên khi logout, DRES trả `401`, đổi tài khoản hoặc app khởi động lại.

### 11.3 Local API

| Method | Endpoint | Mục đích |
|---|---|---|
| POST | `/api/dres/login/default` | Đăng nhập bằng tài khoản mặc định |
| POST | `/api/dres/login` | Đăng nhập tài khoản do người dùng nhập |
| GET | `/api/dres/session` | Khôi phục trạng thái sau reload |
| GET | `/api/dres/evaluations` | Làm mới evaluation |
| GET | `/api/dres/current-task/{evaluation_id}` | Lấy task hiện tại |
| POST | `/api/dres/submit` | Validate và chuyển tiếp submission |
| POST | `/api/dres/logout` | Kết thúc phiên |

### 11.4 Model submit nội bộ

```json
{
  "evaluationId": "...",
  "mode": "kis | qa | trake",
  "videoId": "L21_V001",
  "timeMs": 12400,
  "answer": "màu đỏ",
  "frameIds": [310, 525, 870]
}
```

Backend validate theo `mode`, chuẩn hóa basename và tự tạo payload DRES. Không nhận nguyên chuỗi DRES tùy ý từ frontend.

## 12. Thiết kế frontend

### 12.1 HTML/CSS

- Modal đăng nhập mặc định hoặc tài khoản khác.
- Menu kết nối và chọn evaluation.
- Modal xác nhận KIS/Q&A.
- Ô trả lời Q&A.
- TRAKE Workspace và modal xác nhận riêng.
- Trạng thái loading, disabled, lỗi, hết phiên và verdict.
- Focus bàn phím và bố cục responsive.

### 12.2 JavaScript

- `static/js/api.js`: các lời gọi local DRES API.
- `static/js/dres-session.js` mới: login, logout, evaluation, current task.
- `static/js/submission.js` mới: luồng chung KIS/Q&A và snapshot vị trí.
- `static/js/trake.js` mới: danh sách mốc, sắp xếp và nộp TRAKE.
- `static/js/results.js`: mọi nút card gọi module mới; thêm nút TRAKE.
- `static/js/video-player.js`: Submit dùng `currentTime`; TRAKE thêm frame đang dừng.
- `static/js/main.js`: khởi tạo các module, bỏ DRES token khỏi `localStorage`.
- `static/js/elements.js`: đăng ký DOM element mới.

## 13. Cấu hình

`.env.example` chỉ chứa placeholder:

```env
DRES_BASE_URL=https://eventretrieval.one
DRES_REQUEST_TIMEOUT_SECONDS=10
DRES_DEFAULT_USERNAME=team_197
DRES_DEFAULT_PASSWORD=
```

Trong `.env` cục bộ, đặt `DRES_DEFAULT_PASSWORD` bằng mật khẩu đã cung cấp trong bản kế hoạch cũ.

Quy tắc:

- Không commit `.env`.
- Không trả password mặc định cho frontend.
- Chỉ username mặc định được hiển thị.
- DRES bên ngoài localhost phải dùng HTTPS.

## 14. Nhận diện loại task

- Sau khi chọn evaluation, gọi current-task endpoint.
- Nếu `taskType` ánh xạ rõ sang KIS, Q&A hoặc TRAKE thì tự chọn chế độ.
- Nếu không chắc chắn, bắt buộc người dùng chọn thủ công.
- Chế độ đang dùng phải hiển thị rõ ở header/modal.
- Làm mới task trước khi submit nếu metadata đã cũ.

## 15. Chống nộp trùng

- Khóa nút ngay khi request bắt đầu.
- Tạo task fingerprint từ evaluation ID, task name, group và type.
- Tạo payload fingerprint từ JSON DRES đã chuẩn hóa.
- Chỉ ghi nhận sau response `200` hoặc `202`.
- Chặn cùng payload trong cùng task.
- Task mới cho phép submission mới.
- DRES vẫn là lớp kiểm tra cuối nếu không lấy được task metadata.

## 16. Xử lý response và lỗi

| Trường hợp | Hành vi |
|---|---|
| `200` | Hiển thị verdict và đánh dấu đã gửi |
| `202` | Hiển thị “Đã nhận, đang chờ verdict” |
| `400` | Hiển thị lỗi dữ liệu, giữ nội dung để sửa |
| `401` | Xóa session và yêu cầu login lại |
| `404` | Yêu cầu làm mới evaluation/task |
| `412` | Hiển thị lý do DRES từ chối |
| Timeout/mất mạng | Cho thử lại, không đánh dấu đã gửi |
| Nhấn hai lần | Bỏ qua lần thứ hai khi request đang chạy |

## 17. Kế hoạch kiểm thử

### 17.1 Backend

- [ ] Login mặc định dùng đúng cấu hình backend.
- [ ] Login tài khoản khác không lưu password.
- [ ] Chỉ ưu tiên evaluation ACTIVE.
- [ ] KIS card chuyển đúng frame/fps thành millisecond.
- [ ] KIS video dùng đúng `currentTime` snapshot.
- [ ] `start` và `end` là số nguyên và bằng nhau.
- [ ] Video ID được bỏ phần mở rộng.
- [ ] Q&A tạo đúng chuỗi tiếng Việt.
- [ ] TRAKE tạo đúng chuỗi và giữ thứ tự.
- [ ] TRAKE từ chối nhiều video, frame âm hoặc frame trùng.
- [ ] Xử lý đúng `200`, `202`, `400`, `401`, `404`, `412`.
- [ ] Chống duplicate trong cùng task.
- [ ] Logout xóa session.

### 17.2 Frontend/thủ công

- [ ] Đăng nhập mặc định một chạm.
- [ ] Chuyển sang tài khoản khác và quay lại được.
- [ ] Có thể đổi evaluation và logout.
- [ ] Mọi Submit card mở cùng một modal.
- [ ] Tua video khỏi keyframe ban đầu rồi Submit dùng vị trí mới.
- [ ] Tiến/lùi từng frame rồi Submit cho frame/time đúng.
- [ ] Q&A bắt buộc có câu trả lời và giữ tiếng Việt.
- [ ] Thêm nhiều mốc TRAKE không gửi ngay.
- [ ] Nạp được temporal sequence vào TRAKE Workspace.
- [ ] Sắp xếp, thay thế và xóa mốc TRAKE được.
- [ ] Không trộn hai video trong TRAKE.
- [ ] Nộp TRAKE đúng thứ tự hiển thị.
- [ ] Hiển thị khác nhau cho accepted, pending, rejected, timeout và expired.

### 17.3 Smoke test ngày thi

- [ ] Xác nhận DRES URL chính thức.
- [ ] Login tài khoản mặc định.
- [ ] Chọn evaluation chung kết ACTIVE.
- [ ] Thử KIS từ card.
- [ ] Thử KIS sau khi tua video sang vị trí mới.
- [ ] Thử Q&A bằng tiếng Việt.
- [ ] Thử TRAKE với nhiều frame đã sắp thứ tự.
- [ ] Đối chiếu trực tiếp dữ liệu nhận được trên DRES.
- [ ] Kiểm tra đổi tài khoản và đăng nhập lại.

## 18. Tệp dự kiến thay đổi

| Tệp | Nội dung |
|---|---|
| `app.py` | Thay endpoint mock bằng route DRES thật |
| `services/dres_client.py` | DRES API client |
| `services/dres_session.py` | Session và duplicate store |
| `templates/index.html` | Login, submit modal, TRAKE Workspace |
| `static/js/api.js` | Local DRES API wrapper |
| `static/js/dres-session.js` | Login/evaluation/task state |
| `static/js/submission.js` | KIS/Q&A dùng chung |
| `static/js/trake.js` | Luồng TRAKE riêng |
| `static/js/results.js` | Kết nối các nút card |
| `static/js/video-player.js` | Submit tại vị trí video dừng |
| `static/js/main.js` | Khởi tạo module mới |
| `static/style.css` | Trạng thái login, modal, TRAKE |
| `.env.example` | Cấu hình DRES không có secret |
| `pyproject.toml`, `uv.lock` | Khai báo `httpx` trực tiếp nếu cần |
| `README.md` | Hướng dẫn cấu hình/vận hành |
| `tests/test_dres_client.py` | Test client/session |
| `tests/test_dres_submission.py` | Test payload ba loại bài |

## 19. Thứ tự triển khai

1. [x] Duyệt kế hoạch.
2. [ ] Thêm cấu hình DRES và password mặc định vào `.env` cục bộ.
3. [x] Tạo DRES client và session store.
4. [x] Thay endpoint login/evaluation/submit/logout giả lập.
5. [x] Viết test backend cho ba payload.
6. [x] Làm giao diện login mặc định và đổi tài khoản.
7. [x] Tạo shared handler KIS/Q&A.
8. [x] Chuyển mọi Submit card sang handler mới.
9. [x] Sửa Submit video dùng snapshot `currentTime`.
10. [x] Thêm ô trả lời và preview Q&A.
11. [x] Tạo TRAKE Workspace và module riêng.
12. [x] Thêm thao tác nạp/sắp xếp/thay thế/xóa mốc.
13. [x] Thêm nút Nộp TRAKE và payload cuối.
14. [x] Thêm current-task refresh và chống duplicate.
15. [x] Hoàn thiện lỗi, loading và responsive.
16. [x] Cập nhật README và `.env.example`.
17. [x] Chạy unit test bằng `uv`, kiểm tra cú pháp JavaScript và HTTP smoke test cục bộ.
18. [ ] Kiểm thử trực quan đầy đủ trong trình duyệt (môi trường hiện tại không có browser khả dụng).
19. [ ] Live smoke test chỉ khi được phép dùng DRES chính thức.

Kết quả xác minh ngày 23/09/2026:

- `12/12` unit test đạt cho payload KIS/Q&A/TRAKE, DRES client, session store và Mock DRES.
- Luồng backend thực tế qua Mock DRES đã nhận đúng cả ba payload KIS, Q&A và TRAKE.
- Toàn bộ module JavaScript thay đổi vượt qua `node --check`.
- 8 route DRES được đăng ký thành công khi import ứng dụng.
- `GET /` và `GET /api/dres/session` trả `200` trong HTTP smoke test cục bộ.
- Chưa gọi đăng nhập hoặc submit tới DRES chính thức để tránh tạo tác động ngoài ý muốn.

## 20. Điều kiện hoàn thành

- Không còn login hoặc submit giả lập.
- Đăng nhập nhanh bằng `team_197` hoạt động qua cấu hình backend.
- Người dùng đổi được tài khoản và evaluation.
- Password/session DRES không nằm trong JavaScript hoặc `localStorage`.
- Các nút Submit card hoạt động thống nhất.
- Submit trong video luôn lấy vị trí đang dừng tại thời điểm nhấn.
- KIS/Q&A gửi đúng video basename và millisecond.
- Q&A gửi đúng chuỗi có tiếng Việt.
- TRAKE có luồng chọn nhiều frame riêng và không gửi từng frame.
- TRAKE giữ đúng video và thứ tự semantic keyframe.
- Ba payload đúng tài liệu PDF.
- Có chống double-click và duplicate theo task.
- UI hiển thị đúng verdict/trạng thái DRES.
- Test tự động và thủ công đều đạt.
- Tài liệu cấu hình/vận hành được cập nhật.

## 21. Ghi chú chỉnh sửa và phê duyệt

- [x] Phê duyệt kế hoạch như hiện tại.
- [ ] Cần chỉnh sửa thêm (nếu phát sinh sau smoke test DRES chính thức).

## 22. Kế hoạch kiểm thử giao diện khi DRES chính thức không hoạt động

Mục tiêu: cho phép kiểm tra trực tiếp toàn bộ trải nghiệm đăng nhập và nộp bài mà không cần session thật và không thể gửi nhầm lên DRES chính thức.

1. [x] Tạo Mock DRES độc lập tại `127.0.0.1:19100` với đúng các endpoint login, evaluation, current task, submit và logout.
2. [x] Chỉ cung cấp một evaluation giả `mock-final`; người dùng chọn KIS, Q&A hoặc TRAKE bằng dropdown **Loại submission** trên giao diện.
3. [x] Validate cấu trúc payload tại mock server; payload sai trả `412`, payload đúng trả `200`.
4. [x] Lưu các submission nhận được trong bộ nhớ, tuyệt đối không lưu password.
5. [x] Tạo trang debug để người dùng xem URL, query và JSON body mà ứng dụng đã gửi.
6. [x] Thêm file cấu hình mock riêng, không thay đổi `.env` thật.
7. [x] Viết test tự động cho luồng login → evaluation → submit ba định dạng.
8. [x] Viết hướng dẫn chạy hai server và kịch bản thao tác trên giao diện.

Điều kiện an toàn:

- Mock chỉ lắng nghe trên localhost.
- Cấu hình mock dùng password giả, không chứa credential chính thức.
- Production app chỉ dùng mock khi được khởi động rõ ràng với file cấu hình mock.
- Dừng mock và khởi động lại ứng dụng bình thường sẽ quay về cấu hình DRES thật.

Quy ước cho một evaluation mock:

- Current task dùng loại `MANUAL`, vì vậy không ghi đè dropdown chế độ trên frontend.
- Cùng `evaluationID=mock-final` nhận được cả ba payload KIS, Q&A và TRAKE.
- Mock server tự suy ra loại submission từ `mediaItemName`, tiền tố `QA-` hoặc tiền tố `TR-` rồi mới validate.
