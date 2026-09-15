# Person Search

Hệ thống tìm đúng một người trong camera/video theo kiến trúc trong đề bài:

`ảnh mục tiêu → YOLO person crop → OSNet-AIN embedding`

`camera/video → YOLO11 → BoT-SORT → embedding từng frame → cosine similarity realtime → temporal confirmation`

Giao diện được dựng theo dashboard tham chiếu: tải ảnh mục tiêu, camera/video, bounding box theo Track ID, best match, thống kê, ngưỡng và lịch sử phát hiện.

## Điểm mạnh về độ chính xác

- YOLO11m ở kích thước 960 px thay vì model nano; có thể đổi sang `yolo11x.pt` nếu GPU đủ mạnh.
- BoT-SORT duy trì danh tính qua nhiều frame và xử lý che khuất tốt hơn so sánh từng frame độc lập.
- OSNet-AIN x1.0 dùng checkpoint MSMT17 chuyên cho person Re-ID, không dùng feature ImageNet chung chung.
- Ảnh mục tiêu có flip test-time augmentation. Score của mỗi track được tính trực tiếp từ crop trong frame hiện tại nên phản ánh realtime.
- Phải vượt ngưỡng 2 frame liên tiếp mới gắn nhãn `TARGET`, giảm false positive mà không làm trễ score hiển thị.

> Đây là person Re-ID dựa chủ yếu vào toàn thân/trang phục, không phải nhận dạng khuôn mặt. Nếu hai người mặc giống nhau hoặc mục tiêu thay quần áo, nên thêm face recognition (khi có sự đồng ý và phù hợp quy định riêng tư).

## Cài và chạy trên Windows

Yêu cầu NVIDIA GPU khuyến nghị từ 6–8 GB VRAM. CPU vẫn chạy được nhưng FPS thấp. Dự án dùng Python 3.11/3.12 vì hệ sinh thái PyTorch có thể chưa hỗ trợ bản 3.14 trên máy.

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup.ps1
.\start.ps1
```

Mở `http://127.0.0.1:8000`. Lần chạy đầu ứng dụng tự tải YOLO và checkpoint OSNet-AIN vào thư mục `models/`, vì vậy có thể mất vài phút. Trọng số được `.gitignore` loại khỏi GitHub và sẽ tự tải lại trên máy mới.

`setup.ps1` tự phát hiện NVIDIA và cài wheel PyTorch CUDA 12.8; nếu không có NVIDIA thì dùng CPU. Sau khi cài, dòng cuối cần báo `PyTorch device: GPU` để có tốc độ realtime. Nếu báo CPU dù có NVIDIA, hãy chạy lại `setup.ps1` khi mạng ổn định.

Nếu đã có Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

## Cách dùng

1. Tải một ảnh toàn thân, rõ, ít bị che khuất của người cần tìm.
2. Chọn `Camera` hoặc `Video`. Trình duyệt sẽ gửi frame JPEG tới backend cục bộ qua WebSocket.
3. Bắt đầu với ngưỡng `0.62`. Nếu báo nhầm, tăng từng bước 0.02; nếu bỏ sót, giảm từng bước 0.02.
4. Dùng `Screenshot` để lưu frame cùng bounding box phục vụ báo cáo.

Camera trình duyệt chỉ hoạt động trên `localhost` hoặc HTTPS.

## Cấu hình hiệu năng

Sao chép `.env.example` thành `.env` rồi đặt biến môi trường trước khi chạy (ứng dụng đọc biến môi trường trực tiếp).

| Mục tiêu | `YOLO_MODEL` | `IMAGE_SIZE` |
|---|---:|---:|
| Chính xác cao, GPU mạnh | `yolo11x.pt` | `1280` |
| Cân bằng (mặc định) | `yolo11m.pt` | `960` |
| Máy yếu / CPU | `yolo11n.pt` | `640` |

Ví dụ PowerShell:

```powershell
$env:YOLO_MODEL="yolo11x.pt"
$env:IMAGE_SIZE="1280"
.\start.ps1
```

Đừng chọn ngưỡng chỉ dựa trên vài clip đẹp. Để tối ưu điểm bài tập, hãy tạo validation set gồm cùng người/khác người, nhiều góc, khoảng cách và ánh sáng; chọn ngưỡng cho F1 hoặc mục tiêu false-positive của bài chấm.

Có sẵn công cụ tìm ngưỡng tối ưu trên tập validation:

```powershell
.\.venv\Scripts\python.exe tools\calibrate.py `
  --target data\target.jpg `
  --same data\validation\same `
  --different data\validation\different
```

Mỗi thư mục nên có tối thiểu 20–30 ảnh từ chính camera sẽ trình diễn. Công cụ báo threshold tối ưu, precision, recall, F1 và false-accept rate; đặt kết quả vào `MATCH_THRESHOLD` trong `.env`.

## Cấu trúc

- `app/main.py`: API tải ảnh và WebSocket realtime.
- `app/search.py`: YOLO + BoT-SORT, Re-ID realtime và temporal confirmation.
- `app/reid.py`: OSNet-AIN feature extractor và cosine similarity.
- `tools/calibrate.py`: tìm ngưỡng tốt nhất từ mẫu cùng người/khác người.
- `models/`: nơi tự tải toàn bộ trọng số YOLO/OSNet; Git chỉ lưu `.gitkeep`.
- `static/`: dashboard HTML/CSS/JS, camera/video chạy trong trình duyệt.
- `tests/`: kiểm thử các phần không cần tải model.

## API chính

- `POST /api/target`: ảnh mục tiêu, trả `session_id` và crop được chọn.
- `WS /ws/search/{session_id}`: nhận `{type: "frame", data, threshold}` và trả boxes/stats/best match.
- `GET /api/health`: model, tracker và thiết bị hiện dùng.
